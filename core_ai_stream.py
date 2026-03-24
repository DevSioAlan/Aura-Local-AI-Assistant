"""
core_ai_stream.py
-----------------
Logique principale de génération de texte via l'API Ollama (LLM local).
Gère les appels asynchrones et le streaming des réponses par Server-Sent Events (SSE).

Architecture :
  - OllamaClient   : Encapsule les appels HTTP vers l'API Ollama.
  - StreamHandler  : Transforme les chunks JSON en événements SSE pour le frontend.
  - ContextManager : Gère la fenêtre de contexte (context window) pour ne jamais
                     dépasser la limite de tokens du modèle.
"""

import json
import logging
import time
from typing import Generator, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral"
DEFAULT_VISION_MODEL = "llava"
MAX_CONTEXT_TOKENS = 4096   # limite de tokens conservée en mémoire
REQUEST_TIMEOUT = 120       # secondes


# ---------------------------------------------------------------------------
# Context Window Manager
# ---------------------------------------------------------------------------

class ContextManager:
    """
    Maintient une fenêtre de contexte bornée pour éviter de dépasser
    la limite de tokens du modèle choisi.

    Stratégie : suppression FIFO des messages les plus anciens (hors
    message système) lorsque la limite estimée est dépassée.
    """

    def __init__(self, max_tokens: int = MAX_CONTEXT_TOKENS):
        self.max_tokens = max_tokens

    # Estimation grossière : 1 token ≈ 4 caractères (heuristique standard)
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    def trim(self, messages: list[dict]) -> list[dict]:
        """
        Retourne une version tronquée de la liste de messages respectant
        la limite de tokens.  Le message système (role='system') est
        toujours conservé en tête.
        """
        if not messages:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_msgs = [m for m in messages if m.get("role") != "system"]

        total = sum(
            self._estimate_tokens(m.get("content", ""))
            for m in system_msgs + conv_msgs
        )

        while total > self.max_tokens and len(conv_msgs) > 1:
            removed = conv_msgs.pop(0)
            total -= self._estimate_tokens(removed.get("content", ""))

        return system_msgs + conv_msgs


# ---------------------------------------------------------------------------
# Ollama API Client
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Encapsule les appels REST vers l'API Ollama.

    Méthodes principales :
      - chat_stream()   : génération de texte en streaming (SSE).
      - chat()          : génération de texte classique (réponse complète).
      - list_models()   : liste les modèles disponibles localement.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: dict, stream: bool = False):
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._session.post(
                url,
                json=payload,
                stream=stream,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError as exc:
            logger.error("Connexion à Ollama impossible (%s) : %s", url, exc)
            raise RuntimeError(
                "Le serveur Ollama n'est pas accessible. "
                "Vérifiez qu'Ollama est lancé sur ce poste."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            logger.error("Erreur HTTP Ollama : %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Retourne la liste des modèles installés localement."""
        try:
            resp = self._session.get(
                f"{self.base_url}/api/tags", timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            logger.warning("Impossible de lister les modèles : %s", exc)
            return []

    def chat(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        options: Optional[dict] = None,
    ) -> str:
        """
        Génération de texte (mode synchrone, réponse complète).
        Retourne le contenu textuel du message assistant.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options or {},
        }
        resp = self._post("/api/chat", payload, stream=False)
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def chat_stream(
        self,
        messages: list[dict],
        model: str = DEFAULT_MODEL,
        options: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        """
        Génération de texte en streaming.
        Yield chaque fragment (token) de texte produit par le modèle.
        """
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": options or {},
        }
        resp = self._post("/api/chat", payload, stream=True)

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                logger.warning("Chunk JSON invalide ignoré : %r", raw_line)
                continue

            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

            if chunk.get("done", False):
                break

    def chat_vision_stream(
        self,
        prompt: str,
        image_b64: str,
        model: str = DEFAULT_VISION_MODEL,
    ) -> Generator[str, None, None]:
        """
        Analyse d'image en streaming avec un modèle de vision (ex. LLaVA).

        :param prompt:     Question posée sur l'image.
        :param image_b64:  Image encodée en base64.
        :param model:      Modèle de vision Ollama à utiliser.
        """
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "stream": True,
        }
        resp = self._post("/api/chat", payload, stream=True)

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                chunk = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token

            if chunk.get("done", False):
                break


# ---------------------------------------------------------------------------
# SSE Stream Handler
# ---------------------------------------------------------------------------

class StreamHandler:
    """
    Transforme un générateur de tokens en événements SSE (Server-Sent Events)
    conformes à la spécification W3C EventSource.

    Format de chaque événement :
        data: <json_payload>\\n\\n

    Événements émis :
      - "token"  : fragment de texte en cours de génération.
      - "done"   : fin de la génération (payload : message complet).
      - "error"  : erreur survenue durant la génération.
    """

    @staticmethod
    def _format_event(event_type: str, payload: dict) -> str:
        data = json.dumps({"type": event_type, **payload}, ensure_ascii=False)
        return f"data: {data}\n\n"

    def stream_response(
        self,
        token_generator: Generator[str, None, None],
    ) -> Generator[str, None, None]:
        """
        Consomme un générateur de tokens et produit les événements SSE
        correspondants, puis un événement 'done' final.
        """
        full_response: list[str] = []
        try:
            for token in token_generator:
                full_response.append(token)
                yield self._format_event("token", {"content": token})

            yield self._format_event(
                "done", {"content": "".join(full_response)}
            )
        except RuntimeError as exc:
            logger.error("Erreur durant le streaming : %s", exc)
            yield self._format_event("error", {"message": str(exc)})
        except Exception as exc:
            logger.exception("Erreur inattendue durant le streaming")
            yield self._format_event(
                "error", {"message": "Erreur interne du serveur."}
            )


# ---------------------------------------------------------------------------
# Convenience helpers (utilisés par app.py)
# ---------------------------------------------------------------------------

_ollama = OllamaClient()
_context_mgr = ContextManager()
_stream_handler = StreamHandler()


def generate_sse_stream(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    rag_context: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Point d'entrée principal pour la génération de réponse en streaming SSE.

    :param messages:    Historique de conversation (liste de dicts role/content).
    :param model:       Nom du modèle Ollama à utiliser.
    :param rag_context: Contexte documentaire issu du pipeline RAG (optionnel).
    :returns:           Générateur d'événements SSE prêts à être envoyés au client.
    """
    if rag_context:
        # Injection du contexte RAG dans un message système dédié
        rag_system_msg = {
            "role": "system",
            "content": (
                "Utilise exclusivement le contexte documentaire suivant "
                "pour répondre à la question de l'utilisateur :\n\n"
                f"{rag_context}"
            ),
        }
        # Insertion du contexte RAG avant les messages de conversation
        messages = [rag_system_msg] + [
            m for m in messages if m.get("role") != "system"
        ]

    trimmed = _context_mgr.trim(messages)
    token_gen = _ollama.chat_stream(trimmed, model=model)
    yield from _stream_handler.stream_response(token_gen)


def generate_vision_sse_stream(
    prompt: str,
    image_b64: str,
    model: str = DEFAULT_VISION_MODEL,
) -> Generator[str, None, None]:
    """
    Point d'entrée pour l'analyse d'image en streaming SSE (modèle LLaVA).
    """
    token_gen = _ollama.chat_vision_stream(prompt, image_b64, model=model)
    yield from _stream_handler.stream_response(token_gen)


def get_available_models() -> list[str]:
    """Retourne les modèles LLM disponibles sur le serveur Ollama local."""
    return _ollama.list_models()
