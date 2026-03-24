# 🧠 Aura - Assistant IA Local (Enterprise Portal)

> ⚠️ **Note :** Ce dépôt sert de vitrine technique pour mon portfolio. Pour des raisons de confidentialité, il ne contient pas l'application complète, mais présente **l'architecture et les extraits de code majeurs** (Snippets) illustrant la logique métier que j'ai développée.

## 🎯 Contexte du Projet

Dans les environnements d'entreprise stricts, l'utilisation d'IA Cloud (comme ChatGPT) pose un risque majeur de fuite de données confidentielles. L'objectif de ce projet était de concevoir de zéro un **Portail IA 100% local (Air-gapped)** garantissant une sécurité totale.

## 💡 Mon Rôle & Mes Réalisations

J'ai conçu et développé l'intégralité de l'application (Full-Stack), en relevant plusieurs défis techniques complexes :

* **Garantie de la Confidentialité :** Intégration de l'API Ollama pour faire tourner des modèles LLM (Mistral, Llama 3) localement sur la machine, sans aucune requête externe.
* **Traitement Multimodal (RAG & Vision) :** * Création d'un algorithme Python capable d'extraire et de vectoriser le texte de fichiers PDF locaux pour donner du contexte à l'IA.
  * Intégration de modèles de vision (LLaVA) pour l'analyse d'images uploadées.
* **Architecture Backend Robuste :** Utilisation de Python (Flask) et SQLite3 pour gérer un système de sessions, un historique persistant et une gestion fine de la mémoire (Context Window).
* **Interface Frontend Interactive :** Développement d'une UI en "Glassmorphism" avec gestion des flux de données en temps réel (Server-Sent Events) pour un effet de frappe dynamique (Streaming) fluide, sans framework lourd (Vanilla JS).

## 🛠️ Compétences Techniques Démontrées

* **Langages :** Python 3, JavaScript (ES6), HTML5, CSS3.
* **Backend & BDD :** Flask, SQLite3, Architecture RESTful / SSE.
* **IA & Data :** Prompt Engineering, Ollama API, `pypdf`, `duckduckgo_search`.

## 📂 Contenu de ce dépôt

Vous trouverez dans ce dépôt des extraits illustrant mes méthodes de développement :
* `core_ai_stream.py` : La logique asynchrone de génération de texte et d'appels API.
* `document_parser.py` : L'algorithme de lecture intelligente des documents PDF.
* `frontend_logic.js` : La gestion des Server-Sent Events pour la communication avec l'IA.
