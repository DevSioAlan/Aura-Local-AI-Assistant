/*
 * Vitrine Technique - AURA IA
 * Extrait : Gestion des Server-Sent Events (SSE) côté client.
 * Objectif : Réceptionner le flux de l'IA en temps réel et manipuler le DOM dynamiquement.
 */

function sendMessage(userInput) {
    // [INITIALISATION UI MASQUÉE]

    // Construction dynamique de l'URL avec paramètres (GET)
    const params = new URLSearchParams({
        message: userInput, 
        web: isWebEnabled,
        model: currentModel
    });

    // Ouverture de la connexion Server-Sent Events
    currentEventSource = new EventSource(`/stream_chat?${params.toString()}`);

    currentEventSource.onmessage = (e) => {
        const data = JSON.parse(e.data);
        
        // Routage des événements selon le type retourné par le backend Python
        if (data.type === 'thought') { 
            // Mise à jour de la zone "Réflexion en cours"
            document.getElementById('current-thoughts').innerHTML += `
                <div class="thought-step"><i class="fa-solid fa-check"></i> ${data.content}</div>`; 
        } 
        else if (data.type === 'answer') {
            // Clôture du flux et affichage formaté via Markdown
            currentEventSource.close();
            const ansBox = document.getElementById('current-answer');
            ansBox.innerHTML = marked.parse(data.content);
            hljs.highlightAll(); // Coloration syntaxique du code
            
            scrollToBottom();
        }
    };
}
