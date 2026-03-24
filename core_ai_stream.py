"""
Vitrine Technique - AURA IA
Extrait : Logique de streaming asynchrone (Server-Sent Events) avec l'API Ollama.
Objectif : Gérer le contexte de la conversation, intégrer la recherche web/documents et streamer la réponse.
"""
from flask import Response, stream_with_context
import requests, json, datetime

@app.route('/stream_chat', methods=['GET'])
def stream_chat():
    user_msg = request.args.get('message', '')
    chat_id = request.args.get('chat_id')
    web_mode = request.args.get('web', 'false') == 'true'
    
    # [LOGIQUE DE GESTION DE BASE DE DONNÉES SQLITE MASQUÉE]

    def generate():
        final_context = CONTEXTE_DOCS
        model_to_use = "mistral" # Modèle local
        
        # 1. Traitement multimodal (Images / Web / Documents)
        if DERNIERE_IMAGE:
            yield f"data: {json.dumps({'type': 'thought', 'content': '👁️ Vision...'})}\n\n"
            model_to_use = "llava"
            prompt = f"Description: {user_msg}"
        elif web_mode:
            yield f"data: {json.dumps({'type': 'thought', 'content': '🌍 Recherche...'})}\n\n"
            web_results = search_web(user_msg)
            prompt = f"System: Tu es Aura. Web: {web_results}. Question: {user_msg}"
        else:
            yield f"data: {json.dumps({'type': 'thought', 'content': f'🧠 Analyse...'})}\n\n"
            prompt = f"System: Tu es Aura. Docs: {final_context}. Question: {user_msg}"

        yield f"data: {json.dumps({'type': 'thought', 'content': '✨ Écriture...'})}\n\n"

        # 2. Appel à l'API LLM locale (Ollama)
        payload = { 
            "model": model_to_use, 
            "prompt": prompt, 
            "stream": False, 
            "options": { "num_ctx": 4096, "temperature": 0.7 } 
        }
        
        try:
            resp = requests.post("http://localhost:11434/api/generate", json=payload)
            ai_text = resp.json().get('response', '') if resp.status_code == 200 else "Erreur API"
            
            # [SAUVEGARDE EN BASE DE DONNÉES MASQUÉE]
            
            yield f"data: {json.dumps({'type': 'answer', 'content': ai_text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'answer', 'content': f'Erreur: {e}'})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')
