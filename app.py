"""
app.py
------
Serveur Flask — Portail IA local Aura.
Expose les endpoints REST et SSE utilisés par le frontend.

Endpoints :
  POST /api/chat            → Génération de réponse en streaming SSE.
  POST /api/upload/pdf      → Indexation d'un document PDF (pipeline RAG).
  POST /api/upload/image    → (Réservé — les images sont envoyées dans /api/chat).
  GET  /api/models          → Liste des modèles Ollama disponibles.
  POST /api/session/new     → Création d'une nouvelle session.
  POST /api/session/clear   → Réinitialisation de l'historique d'une session.
  GET  /                    → Sert l'interface HTML principale.
"""

import json
import logging
import os
import sqlite3
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file, session

from core_ai_stream import (
    DEFAULT_MODEL,
    generate_sse_stream,
    generate_vision_sse_stream,
    get_available_models,
)
from document_parser import get_rag_context, process_uploaded_pdf, reset_rag_index

# ---------------------------------------------------------------------------
# Application & configuration
# ---------------------------------------------------------------------------

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(32))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
)
logger = logging.getLogger(__name__)

ALLOWED_PDF_EXTENSIONS = {".pdf"}
MAX_PDF_SIZE_MB = 50
MAX_MESSAGE_LENGTH = 8000

# Chemin vers la base SQLite (stockée localement, jamais en cloud)
DB_PATH = Path(os.environ.get("AURA_DB_PATH", "aura_history.db"))


# ---------------------------------------------------------------------------
# Persistance SQLite — gestion des sessions et de l'historique
# ---------------------------------------------------------------------------

def _init_db() -> None:
    """Crée les tables nécessaires si elles n'existent pas encore."""
    with _db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role      TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
                content   TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)"
        )


@contextmanager
def _db():
    """Gestionnaire de contexte pour une connexion SQLite thread-safe."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_session(session_id: str) -> None:
    """Insère la session dans la BDD si elle n'existe pas."""
    with _db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions(id) VALUES (?)", (session_id,)
        )


def _get_history(session_id: str) -> list[dict]:
    """Charge l'historique de conversation depuis SQLite."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def _append_to_history(session_id: str, role: str, content: str) -> None:
    """Ajoute un message à l'historique persistant."""
    _ensure_session(session_id)
    with _db() as conn:
        conn.execute(
            "INSERT INTO messages(session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def _clear_history(session_id: str) -> None:
    """Supprime tous les messages d'une session."""
    with _db() as conn:
        conn.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_session_id() -> str:
    """Retourne l'identifiant de session courant ou en crée un nouveau."""
    sid = session.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())
        session["session_id"] = sid
        _ensure_session(sid)
    return sid


# ---------------------------------------------------------------------------
# Routes principales
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/static/frontend_logic.js")
def serve_frontend_js():
    """Sert le fichier frontend_logic.js depuis la racine du projet."""
    js_path = Path(__file__).parent / "frontend_logic.js"
    return send_file(js_path, mimetype="application/javascript")


# ---------------------------------------------------------------------------
# API — Chat (SSE streaming)
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Reçoit un message utilisateur et répond en streaming SSE.
    Corps JSON attendu :
      - message   (str, requis)
      - model     (str, optionnel)
      - session_id(str, optionnel)
      - image     (str base64, optionnel — déclenche le mode Vision)
    """
    data = request.get_json(silent=True) or {}

    user_message: str = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Le champ 'message' est requis."}), 400
    if len(user_message) > MAX_MESSAGE_LENGTH:
        return jsonify({"error": "Message trop long."}), 400

    model: str = data.get("model") or DEFAULT_MODEL
    image_b64: str | None = data.get("image")

    session_id = data.get("session_id") or _get_or_create_session_id()
    _append_to_history(session_id, "user", user_message)

    def event_stream():
        full_response_parts: list[str] = []

        try:
            if image_b64:
                # Mode Vision (LLaVA)
                gen = generate_vision_sse_stream(
                    prompt=user_message,
                    image_b64=image_b64,
                )
            else:
                # Mode texte avec RAG optionnel
                rag_context = get_rag_context(user_message)
                history = _get_history(session_id)
                gen = generate_sse_stream(
                    messages=history,
                    model=model,
                    rag_context=rag_context or None,
                )

            for chunk in gen:
                # Accumulation pour sauvegarder la réponse finale
                if '"type": "token"' in chunk:
                    try:
                        evt = json.loads(chunk.replace("data: ", "").strip())
                        full_response_parts.append(evt.get("content", ""))
                    except (json.JSONDecodeError, ValueError):
                        pass
                yield chunk

        except Exception:
            logger.exception("Erreur dans le flux SSE")
            error_event = json.dumps(
                {"type": "error", "message": "Erreur interne du serveur."}
            )
            yield f"data: {error_event}\n\n"
        finally:
            full_response = "".join(full_response_parts)
            if full_response:
                _append_to_history(session_id, "assistant", full_response)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# API — Upload PDF (RAG)
# ---------------------------------------------------------------------------

@app.route("/api/upload/pdf", methods=["POST"])
def api_upload_pdf():
    """
    Reçoit un fichier PDF multipart et l'indexe dans le pipeline RAG.
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Aucun fichier reçu."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"success": False, "error": "Nom de fichier manquant."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_PDF_EXTENSIONS:
        return jsonify(
            {"success": False, "error": "Seuls les fichiers PDF sont acceptés."}
        ), 400

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = process_uploaded_pdf(tmp_path, original_name=file.filename)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    status_code = 200 if result["success"] else 422
    return jsonify(result), status_code


# ---------------------------------------------------------------------------
# API — Modèles disponibles
# ---------------------------------------------------------------------------

@app.route("/api/models", methods=["GET"])
def api_models():
    """Retourne la liste des modèles Ollama installés localement."""
    models = get_available_models()
    return jsonify({"models": models})


# ---------------------------------------------------------------------------
# API — Gestion des sessions
# ---------------------------------------------------------------------------

@app.route("/api/session/new", methods=["POST"])
def api_session_new():
    sid = str(uuid.uuid4())
    session["session_id"] = sid
    _ensure_session(sid)
    return jsonify({"session_id": sid})


@app.route("/api/session/clear", methods=["POST"])
def api_session_clear():
    data = request.get_json(silent=True) or {}
    sid = data.get("session_id") or session.get("session_id")
    if sid:
        _clear_history(sid)
    reset_rag_index()
    return jsonify({"success": True})


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

_init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

