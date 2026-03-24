"""
document_parser.py
-------------------
Algorithme de lecture intelligente de documents PDF pour le pipeline RAG
(Retrieval-Augmented Generation).

Fonctionnement :
  1. Extraction du texte brut depuis le PDF (via pypdf).
  2. Découpage en chunks sémantiques (chunking).
  3. Vectorisation de chaque chunk (TF-IDF léger, sans dépendance externe lourde).
  4. Recherche par similarité cosinus pour retrouver les passages les plus
     pertinents par rapport à une requête utilisateur.

Ce module est intentionnellement autonome (pas de base vectorielle externe)
afin de garantir le fonctionnement 100 % hors-ligne (air-gapped).
"""

import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pypdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE = 500        # taille cible d'un chunk en mots
CHUNK_OVERLAP = 50      # chevauchement entre chunks consécutifs (en mots)
TOP_K_CHUNKS = 4        # nombre de chunks renvoyés lors d'une recherche
MAX_FILE_SIZE_MB = 50   # limite de sécurité sur la taille des fichiers

# Pré-compilation des patterns regex pour les performances
_TOKEN_PATTERN = re.compile(r"\b\w+\b")


# ---------------------------------------------------------------------------
# Utilitaires texte
# ---------------------------------------------------------------------------

def _clean_text(raw: str) -> str:
    """Normalise le texte extrait (suppression des sauts de ligne superflus, etc.)."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Fusion des coupures de mots en fin de ligne (ex. "trai-\nter" → "traiter")
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Réduction des séquences de blancs internes
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Réduction des lignes vides multiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    """Découpe le texte en tokens (mots) minuscules sans ponctuation."""
    return _TOKEN_PATTERN.findall(text.lower())


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Divise le texte en chunks de taille fixe (en mots) avec chevauchement
    pour préserver le contexte aux frontières.

    :param text:        Texte complet à découper.
    :param chunk_size:  Nombre cible de mots par chunk.
    :param overlap:     Nombre de mots de chevauchement entre chunks.
    :returns:           Liste de chunks textuels.
    """
    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------------------------
# Vectorisation TF-IDF (implémentation légère, sans sklearn)
# ---------------------------------------------------------------------------

class TfidfVectorizer:
    """
    Vectoriseur TF-IDF minimaliste ne nécessitant aucune bibliothèque ML.
    Utilisé pour la similarité sémantique entre la requête et les chunks.
    """

    def __init__(self):
        self._idf: dict[str, float] = {}
        self._vocab: list[str] = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, documents: list[str]) -> "TfidfVectorizer":
        """
        Calcule l'IDF de chaque terme à partir du corpus de documents.
        """
        n_docs = len(documents)
        if n_docs == 0:
            return self

        doc_freq: Counter = Counter()
        for doc in documents:
            terms = set(_tokenize(doc))
            doc_freq.update(terms)

        self._idf = {
            # Formule IDF lissée : log((N+1)/(df+1)) + 1
            # Le +1 dans le numérateur et le dénominateur évite la division par zéro
            # et atténue l'effet des termes très rares. Le +1 final empêche un IDF
            # nul pour les termes présents dans tous les documents.
            term: math.log((n_docs + 1) / (freq + 1)) + 1.0
            for term, freq in doc_freq.items()
        }
        self._vocab = sorted(self._idf.keys())
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, documents: list[str]) -> list[list[float]]:
        """
        Calcule le vecteur TF-IDF normalisé pour chaque document.
        """
        vectors = []
        for doc in documents:
            tokens = _tokenize(doc)
            if not tokens:
                vectors.append([0.0] * len(self._vocab))
                continue

            tf = Counter(tokens)
            n_tokens = len(tokens)

            vec = [
                (tf.get(term, 0) / n_tokens) * self._idf.get(term, 0.0)
                for term in self._vocab
            ]
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])

        return vectors

    def fit_transform(self, documents: list[str]) -> list[list[float]]:
        return self.fit(documents).transform(documents)


# ---------------------------------------------------------------------------
# Similarité cosinus
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calcule la similarité cosinus entre deux vecteurs."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# DocumentParser (point d'entrée principal)
# ---------------------------------------------------------------------------

class DocumentParser:
    """
    Analyse un ou plusieurs documents PDF et expose une interface de
    recherche sémantique pour le pipeline RAG.

    Utilisation :
        parser = DocumentParser()
        parser.load_pdf(Path("rapport.pdf"))
        context = parser.retrieve("Quels sont les résultats financiers ?")
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        top_k: int = TOP_K_CHUNKS,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self._chunks: list[str] = []
        self._chunk_sources: list[str] = []   # nom de fichier source par chunk
        self._vectorizer = TfidfVectorizer()
        self._chunk_vectors: list[list[float]] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Chargement de documents
    # ------------------------------------------------------------------

    def load_pdf(self, pdf_path: Path, source_name: Optional[str] = None) -> int:
        """
        Charge un fichier PDF, extrait son texte, le découpe en chunks et
        met à jour l'index vectoriel.

        :param pdf_path:    Chemin absolu vers le fichier PDF.
        :param source_name: Libellé source affiché dans les métadonnées
                            (défaut : nom du fichier).
        :returns:           Nombre de chunks ajoutés.
        :raises FileNotFoundError: Si le fichier est introuvable.
        :raises ValueError:        Si le fichier est trop volumineux ou illisible.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF introuvable : {pdf_path}")

        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise ValueError(
                f"Le fichier dépasse la limite de {MAX_FILE_SIZE_MB} Mo "
                f"({size_mb:.1f} Mo)."
            )

        source_name = source_name or pdf_path.name
        raw_text = self._extract_text(pdf_path)
        clean = _clean_text(raw_text)

        new_chunks = _split_into_chunks(clean, self.chunk_size, self.chunk_overlap)
        if not new_chunks:
            logger.warning("Aucun contenu extrait de : %s", pdf_path)
            return 0

        self._chunks.extend(new_chunks)
        self._chunk_sources.extend([source_name] * len(new_chunks))
        self._refit()

        logger.info(
            "%d chunks indexés depuis '%s' (total : %d)",
            len(new_chunks), source_name, len(self._chunks),
        )
        return len(new_chunks)

    @staticmethod
    def _extract_text(pdf_path: Path) -> str:
        """Extrait le texte de toutes les pages d'un PDF via pypdf."""
        text_parts: list[str] = []
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                    text_parts.append(text)
                except Exception as exc:
                    logger.warning(
                        "Impossible d'extraire la page %d : %s", page_num, exc
                    )
        except Exception as exc:
            raise ValueError(f"Erreur lors de la lecture du PDF : {exc}") from exc

        return "\n\n".join(text_parts)

    # ------------------------------------------------------------------
    # Indexation vectorielle
    # ------------------------------------------------------------------

    def _refit(self) -> None:
        """Recalcule l'index TF-IDF sur l'ensemble des chunks courants."""
        if self._chunks:
            self._chunk_vectors = self._vectorizer.fit_transform(self._chunks)
            self._fitted = True

    # ------------------------------------------------------------------
    # Recherche sémantique (RAG Retrieval)
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Recherche les chunks les plus pertinents pour une requête donnée
        et retourne un contexte textuel consolidé.

        :param query:  Question ou requête utilisateur.
        :param top_k:  Nombre de chunks à retourner (défaut : self.top_k).
        :returns:      Texte concaténé des chunks les plus pertinents,
                       prêt à être injecté dans le prompt système.
        """
        if not self._fitted or not self._chunks:
            return ""

        k = top_k if top_k is not None else self.top_k
        query_vec = self._vectorizer.transform([query])[0]

        scored = [
            (idx, _cosine_similarity(query_vec, chunk_vec))
            for idx, chunk_vec in enumerate(self._chunk_vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, score in scored[:k] if score > 0]

        if not top_indices:
            return ""

        # Reconstruction ordonnée pour la cohérence de lecture
        top_indices.sort()
        context_parts = [
            f"[Source : {self._chunk_sources[i]}]\n{self._chunks[i]}"
            for i in top_indices
        ]
        return "\n\n---\n\n".join(context_parts)

    # ------------------------------------------------------------------
    # Statistiques
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Retourne des informations sur le corpus indexé."""
        return {
            "total_chunks": len(self._chunks),
            "sources": list(dict.fromkeys(self._chunk_sources)),
            "fitted": self._fitted,
        }

    def clear(self) -> None:
        """Supprime tous les chunks et réinitialise l'index."""
        self._chunks = []
        self._chunk_sources = []
        self._chunk_vectors = []
        self._fitted = False
        self._vectorizer = TfidfVectorizer()


# ---------------------------------------------------------------------------
# Interface de haut niveau (utilisée par app.py)
# ---------------------------------------------------------------------------

_parser = DocumentParser()


def process_uploaded_pdf(file_path: str, original_name: str) -> dict:
    """
    Charge un PDF uploadé par l'utilisateur dans l'index RAG global.

    :param file_path:     Chemin temporaire du fichier sur le serveur.
    :param original_name: Nom d'origine du fichier (pour les métadonnées).
    :returns:             Dictionnaire de résultat (success, chunks_added, stats).
    """
    try:
        n = _parser.load_pdf(Path(file_path), source_name=original_name)
        return {"success": True, "chunks_added": n, "stats": _parser.stats()}
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Erreur chargement PDF : %s", exc)
        return {"success": False, "error": str(exc)}


def get_rag_context(query: str, top_k: int = TOP_K_CHUNKS) -> str:
    """
    Retourne le contexte documentaire pertinent pour une requête donnée.
    Retourne une chaîne vide si aucun document n'est indexé.
    """
    return _parser.retrieve(query, top_k=top_k)


def reset_rag_index() -> None:
    """Vide l'index RAG (ex. : en début de nouvelle session)."""
    _parser.clear()
