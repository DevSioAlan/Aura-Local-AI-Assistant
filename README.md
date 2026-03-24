# 🧠 Aura — Assistant IA Local (Enterprise Portal)

> ⚠️ **Note :** Ce dépôt sert de vitrine technique pour mon portfolio. Pour des raisons de confidentialité, il ne contient pas l'application complète, mais présente **l'architecture et les extraits de code majeurs** (Snippets) illustrant la logique métier que j'ai développée.

## 🎯 Contexte du Projet

Dans les environnements d'entreprise stricts, l'utilisation d'IA Cloud (comme ChatGPT) pose un risque majeur de fuite de données confidentielles. L'objectif de ce projet était de concevoir de zéro un **Portail IA 100 % local (Air-gapped)** garantissant une sécurité totale.

## 💡 Mon Rôle & Mes Réalisations

J'ai conçu et développé l'intégralité de l'application (Full-Stack), en relevant plusieurs défis techniques complexes :

- **Garantie de la Confidentialité :** Intégration de l'API Ollama pour faire tourner des modèles LLM (Mistral, Llama 3) localement sur la machine, sans aucune requête externe.
- **Traitement Multimodal (RAG & Vision) :**
  - Création d'un algorithme Python capable d'extraire et de vectoriser le texte de fichiers PDF locaux pour donner du contexte à l'IA.
  - Intégration de modèles de vision (LLaVA) pour l'analyse d'images uploadées.
- **Architecture Backend Robuste :** Utilisation de Python (Flask) et SQLite3 pour gérer un système de sessions, un historique persistant des conversations et une gestion fine de la mémoire (Context Window).
- **Interface Frontend Interactive :** Développement d'une UI en "Glassmorphism" avec gestion des flux de données en temps réel (Server-Sent Events) pour un effet de frappe dynamique (Streaming) fluide, sans framework lourd (Vanilla JS).

## 🛠️ Compétences Techniques Démontrées

| Catégorie       | Technologies                                              |
|-----------------|-----------------------------------------------------------|
| **Langages**    | Python 3, JavaScript (ES6+), HTML5, CSS3                  |
| **Backend**     | Flask, SQLite3, Architecture RESTful / SSE                |
| **IA & Data**   | Prompt Engineering, Ollama API, `pypdf`, TF-IDF vectorisation |
| **Sécurité**    | Déploiement Air-gapped, aucune dépendance réseau externe  |

## 📂 Contenu de ce dépôt

| Fichier                  | Description                                                                                           |
|--------------------------|-------------------------------------------------------------------------------------------------------|
| `core_ai_stream.py`      | Logique asynchrone de génération de texte (Ollama API, SSE streaming, gestion du context window).     |
| `document_parser.py`     | Algorithme de lecture intelligente des PDF : extraction, chunking, vectorisation TF-IDF et RAG.       |
| `frontend_logic.js`      | Gestion des Server-Sent Events, effet typewriter, upload PDF/image, rendu Markdown — Vanilla JS.      |
| `app.py`                 | Serveur Flask : endpoints REST/SSE, gestion des sessions, intégration RAG et Vision.                  |
| `requirements.txt`       | Dépendances Python du projet.                                                                         |
| `templates/index.html`   | Interface HTML principale (Glassmorphism).                                                            |
| `static/css/style.css`   | Feuille de style complète (variables CSS, animations, responsive).                                    |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Navigateur (Client)               │
│  Vanilla JS · SSE · Glassmorphism UI                │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP / SSE
┌───────────────────────▼─────────────────────────────┐
│               Flask Backend (app.py)                │
│  Sessions · Historique · Context Window Manager     │
├──────────────┬──────────────────────────────────────┤
│ RAG Pipeline │  document_parser.py                  │
│              │  (pypdf · TF-IDF · cosine similarity)│
├──────────────┴──────────────────────────────────────┤
│ AI Streaming │  core_ai_stream.py                   │
│              │  (OllamaClient · StreamHandler · SSE)│
└──────────────┬──────────────────────────────────────┘
               │ API locale (http://localhost:11434)
┌──────────────▼──────────────────────────────────────┐
│              Ollama (LLM local)                     │
│  Mistral · Llama 3 · LLaVA (Vision)                 │
└─────────────────────────────────────────────────────┘
```

## 🚀 Lancement rapide

> **Pré-requis :** [Ollama](https://ollama.com/) installé et au moins un modèle téléchargé (`ollama pull mistral`).

```bash
# 1. Installer les dépendances Python
pip install -r requirements.txt

# 2. Lancer le serveur Flask
python app.py

# 3. Ouvrir le portail dans votre navigateur
# http://localhost:5000
```

## 🔒 Garantie de confidentialité

- **Aucune requête externe** : tous les modèles LLM s'exécutent sur la machine locale via Ollama.
- **Aucune télémétrie** : les conversations et documents ne quittent jamais l'infrastructure.
- **Données en mémoire uniquement** : l'historique de session est géré en RAM (pas de cloud, pas de logs distants).
