"""
Vitrine Technique - AURA IA
Extrait : Parser de documents (RAG simplifié).
Objectif : Lire le contenu de fichiers PDF locaux pour construire un contexte de prompt.
"""
from pypdf import PdfReader
import glob, os

def lire_documents(dossier_docs):
    content = ""
    # Scan automatique des fichiers PDF uploadés
    for f in glob.glob(os.path.join(dossier_docs, "*.pdf")):
        try:
            reader = PdfReader(f)
            text = ""
            for i, p in enumerate(reader.pages): 
                if i > 1: break # Optimisation : Limitation aux 2 premières pages
                text += p.extract_text() + "\n"
            # Formatage du contexte pour le LLM
            content += f"\n[DOC: {os.path.basename(f)}]\n{text[:1500]}..." 
        except Exception as e: 
            pass
    return content
