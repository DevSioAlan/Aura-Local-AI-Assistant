/**
 * frontend_logic.js
 * ------------------
 * Gestion des interactions avec l'IA via Server-Sent Events (SSE).
 * Interface Glassmorphism — Vanilla JS (ES6+), aucun framework externe.
 *
 * Fonctionnalités :
 *   - Connexion SSE vers le backend Flask pour le streaming des réponses.
 *   - Effet de frappe dynamique (typewriter) sur les tokens reçus.
 *   - Gestion du cycle de vie de la conversation (envoi, réponse, erreur).
 *   - Upload de fichiers PDF pour le pipeline RAG.
 *   - Upload d'images pour l'analyse Vision (LLaVA).
 *   - Rendu Markdown léger (gras, italique, code inline, blocs de code).
 */

// ============================================================
// Configuration
// ============================================================

const API = {
  CHAT: "/api/chat",
  UPLOAD_PDF: "/api/upload/pdf",
  UPLOAD_IMAGE: "/api/upload/image",
  MODELS: "/api/models",
  SESSION_NEW: "/api/session/new",
  SESSION_CLEAR: "/api/session/clear",
};

// ============================================================
// État global de l'application
// ============================================================

const state = {
  /** @type {EventSource|null} */
  currentStream: null,
  isStreaming: false,
  currentModel: "mistral",
  sessionId: null,
  /** @type {AbortController|null} */
  abortController: null,
};

// ============================================================
// Sélecteurs DOM (peuplés une fois le DOM prêt)
// ============================================================

let dom = {};

function initDomRefs() {
  dom = {
    chatMessages: document.getElementById("chat-messages"),
    userInput: document.getElementById("user-input"),
    sendButton: document.getElementById("send-btn"),
    stopButton: document.getElementById("stop-btn"),
    modelSelect: document.getElementById("model-select"),
    pdfUploadInput: document.getElementById("pdf-upload"),
    imageUploadInput: document.getElementById("image-upload"),
    uploadStatus: document.getElementById("upload-status"),
    clearSessionBtn: document.getElementById("clear-session-btn"),
    charCounter: document.getElementById("char-counter"),
  };
}

// ============================================================
// Rendu Markdown minimaliste
// ============================================================

/**
 * Convertit un texte brut (Markdown partiel) en HTML sécurisé.
 * Supporte : blocs de code, code inline, gras, italique, listes.
 *
 * @param {string} raw - Texte brut à convertir.
 * @returns {string} HTML échappé et formaté.
 */
function renderMarkdown(raw) {
  // Échappement des caractères HTML dangereux (sécurité XSS)
  let html = raw
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Blocs de code (``` ... ```)
  html = html.replace(
    /```(\w*)\n?([\s\S]*?)```/g,
    (_, lang, code) =>
      `<pre><code class="language-${lang || "plaintext"}">${code.trim()}</code></pre>`
  );

  // Code inline (`...`)
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Gras (**...**)
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Italique (*...*)
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Listes non ordonnées (regroupe toutes les lignes "- ..." consécutives)
  html = html.replace(/(<li>[\s\S]*?<\/li>(\s*<li>[\s\S]*?<\/li>)*)/g, "<ul>$1</ul>");

  // Sauts de ligne
  html = html.replace(/\n/g, "<br>");

  return html;
}

// ============================================================
// Gestion des messages dans l'interface
// ============================================================

/**
 * Ajoute un message utilisateur dans la zone de chat.
 *
 * @param {string} text - Contenu textuel du message.
 */
function appendUserMessage(text) {
  const wrapper = document.createElement("div");
  wrapper.className = "message user-message";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  // Texte brut pour les messages utilisateur (pas de Markdown)
  bubble.textContent = text;

  wrapper.appendChild(bubble);
  dom.chatMessages.appendChild(wrapper);
  scrollToBottom();
}

/**
 * Crée un conteneur de message assistant vide et retourne
 * une référence vers l'élément de bulle pour y injecter les tokens.
 *
 * @returns {{ wrapper: HTMLElement, bubble: HTMLElement }}
 */
function createAssistantMessagePlaceholder() {
  const wrapper = document.createElement("div");
  wrapper.className = "message assistant-message";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.innerHTML = '<span class="typing-cursor">▍</span>';

  wrapper.appendChild(bubble);
  dom.chatMessages.appendChild(wrapper);
  scrollToBottom();

  return { wrapper, bubble };
}

/**
 * Ajoute un message d'erreur dans la zone de chat.
 *
 * @param {string} message - Description de l'erreur.
 */
function appendErrorMessage(message) {
  const wrapper = document.createElement("div");
  wrapper.className = "message error-message";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = `⚠️ ${message}`;

  wrapper.appendChild(bubble);
  dom.chatMessages.appendChild(wrapper);
  scrollToBottom();
}

/** Fait défiler la zone de chat jusqu'au dernier message. */
function scrollToBottom() {
  dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

// ============================================================
// Streaming SSE
// ============================================================

/**
 * Ouvre un flux SSE vers le backend et traite les tokens reçus
 * pour les afficher en temps réel (effet typewriter).
 *
 * @param {string} userMessage - Message de l'utilisateur à envoyer.
 * @param {string|null} imageB64 - Image encodée en base64 (optionnel).
 */
async function sendMessageWithStream(userMessage, imageB64 = null) {
  if (state.isStreaming) return;

  setStreamingState(true);
  appendUserMessage(userMessage);

  const { bubble } = createAssistantMessagePlaceholder();
  let accumulatedText = "";

  try {
    state.abortController = new AbortController();

    const payload = {
      message: userMessage,
      model: state.currentModel,
      session_id: state.sessionId,
    };
    if (imageB64) {
      payload.image = imageB64;
    }

    const response = await fetch(API.CHAT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: state.abortController.signal,
    });

    if (!response.ok) {
      throw new Error(`Erreur serveur : ${response.status} ${response.statusText}`);
    }

    // Lecture du flux SSE via ReadableStream
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Traitement des lignes SSE complètes ("data: ...\n\n")
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";

      for (const eventBlock of events) {
        const dataLine = eventBlock
          .split("\n")
          .find((line) => line.startsWith("data: "));

        if (!dataLine) continue;

        let parsed;
        try {
          parsed = JSON.parse(dataLine.slice(6));
        } catch {
          continue;
        }

        if (parsed.type === "token") {
          accumulatedText += parsed.content;
          // Mise à jour en direct de la bulle avec rendu Markdown partiel
          bubble.innerHTML =
            renderMarkdown(accumulatedText) +
            '<span class="typing-cursor">▍</span>';
          scrollToBottom();
        } else if (parsed.type === "done") {
          // Rendu final propre (sans curseur)
          bubble.innerHTML = renderMarkdown(parsed.content || accumulatedText);
          scrollToBottom();
        } else if (parsed.type === "error") {
          bubble.innerHTML = `<span class="error-text">⚠️ ${parsed.message}</span>`;
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      // Arrêt volontaire par l'utilisateur
      if (accumulatedText) {
        bubble.innerHTML =
          renderMarkdown(accumulatedText) +
          ' <span class="stopped-badge">[arrêté]</span>';
      } else {
        bubble.remove();
      }
    } else {
      console.error("[Aura] Erreur de streaming :", err);
      bubble.remove();
      appendErrorMessage(err.message || "Impossible de contacter le serveur.");
    }
  } finally {
    setStreamingState(false);
    state.abortController = null;
  }
}

// ============================================================
// Contrôle du mode streaming (UI)
// ============================================================

/**
 * Active ou désactive l'interface selon l'état de génération.
 *
 * @param {boolean} streaming - true pendant la génération, false sinon.
 */
function setStreamingState(streaming) {
  state.isStreaming = streaming;
  dom.sendButton.disabled = streaming;
  dom.userInput.disabled = streaming;
  dom.stopButton.style.display = streaming ? "inline-flex" : "none";
  dom.sendButton.style.display = streaming ? "none" : "inline-flex";
}

/** Arrête le flux SSE en cours (déclenchée par le bouton "Stop"). */
function stopStreaming() {
  if (state.abortController) {
    state.abortController.abort();
  }
}

// ============================================================
// Upload PDF (RAG)
// ============================================================

/**
 * Envoie un fichier PDF au backend pour l'indexation RAG.
 *
 * @param {File} file - Fichier PDF sélectionné par l'utilisateur.
 */
async function uploadPdf(file) {
  showUploadStatus("Chargement du document en cours…", "info");

  const formData = new FormData();
  formData.append("file", file);

  try {
    const resp = await fetch(API.UPLOAD_PDF, {
      method: "POST",
      body: formData,
    });
    const data = await resp.json();

    if (data.success) {
      showUploadStatus(
        `✅ Document indexé : ${data.chunks_added} segments (total : ${data.stats.total_chunks}).`,
        "success"
      );
    } else {
      showUploadStatus(`❌ Erreur : ${data.error}`, "error");
    }
  } catch (err) {
    console.error("[Aura] Erreur upload PDF :", err);
    showUploadStatus("❌ Impossible d'envoyer le document.", "error");
  }
}

// ============================================================
// Upload Image (Vision)
// ============================================================

/**
 * Lit un fichier image sélectionné et le convertit en base64
 * pour l'envoyer avec le prochain message utilisateur.
 *
 * @param {File} file - Fichier image sélectionné.
 * @returns {Promise<string>} Image encodée en base64 (sans préfixe data URL).
 */
function readImageAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      // Suppression du préfixe "data:image/...;base64,"
      const b64 = e.target.result.split(",")[1];
      resolve(b64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// ============================================================
// Gestion des sessions
// ============================================================

/**
 * Démarre une nouvelle session de conversation.
 * Vide l'historique visuel et demande une réinitialisation au backend.
 */
async function clearSession() {
  try {
    await fetch(API.SESSION_CLEAR, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
  } catch (err) {
    console.warn("[Aura] Impossible de réinitialiser la session :", err);
  }

  dom.chatMessages.innerHTML = "";
  appendSystemNotice("Nouvelle conversation démarrée.");
}

/**
 * Affiche une notice système dans la zone de chat.
 *
 * @param {string} text - Texte de la notice.
 */
function appendSystemNotice(text) {
  const notice = document.createElement("div");
  notice.className = "system-notice";
  notice.textContent = text;
  dom.chatMessages.appendChild(notice);
  scrollToBottom();
}

// ============================================================
// Chargement des modèles disponibles
// ============================================================

/**
 * Récupère la liste des modèles Ollama installés et peuple le sélecteur.
 */
async function loadAvailableModels() {
  try {
    const resp = await fetch(API.MODELS);
    const data = await resp.json();
    const models = data.models || [];

    dom.modelSelect.innerHTML = "";
    if (models.length === 0) {
      const opt = document.createElement("option");
      opt.value = "mistral";
      opt.textContent = "mistral (défaut)";
      dom.modelSelect.appendChild(opt);
      return;
    }

    for (const model of models) {
      const opt = document.createElement("option");
      opt.value = model;
      opt.textContent = model;
      if (model === state.currentModel) opt.selected = true;
      dom.modelSelect.appendChild(opt);
    }
  } catch (err) {
    console.warn("[Aura] Impossible de charger les modèles :", err);
  }
}

// ============================================================
// Affichage du statut d'upload
// ============================================================

/**
 * @param {string} message - Texte à afficher.
 * @param {"info"|"success"|"error"} type - Niveau du message.
 */
function showUploadStatus(message, type) {
  if (!dom.uploadStatus) return;
  dom.uploadStatus.textContent = message;
  dom.uploadStatus.className = `upload-status upload-status--${type}`;
  dom.uploadStatus.style.display = "block";

  if (type === "success") {
    setTimeout(() => {
      dom.uploadStatus.style.display = "none";
    }, 5000);
  }
}

// ============================================================
// Initialisation & Event Listeners
// ============================================================

/**
 * Point d'entrée principal — appelé au chargement du DOM.
 */
async function init() {
  initDomRefs();
  await loadAvailableModels();

  // ── Sélection du modèle ──────────────────────────────────────
  dom.modelSelect?.addEventListener("change", (e) => {
    state.currentModel = e.target.value;
  });

  // ── Compteur de caractères ───────────────────────────────────
  dom.userInput?.addEventListener("input", () => {
    const len = dom.userInput.value.length;
    if (dom.charCounter) {
      dom.charCounter.textContent = `${len} / 4000`;
      dom.charCounter.classList.toggle("char-counter--warn", len > 3500);
    }
  });

  // ── Envoi avec Entrée (Shift+Entrée = saut de ligne) ─────────
  dom.userInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // ── Bouton Envoyer ────────────────────────────────────────────
  dom.sendButton?.addEventListener("click", handleSend);

  // ── Bouton Stop ───────────────────────────────────────────────
  dom.stopButton?.addEventListener("click", stopStreaming);

  // ── Bouton Réinitialiser session ──────────────────────────────
  dom.clearSessionBtn?.addEventListener("click", clearSession);

  // ── Upload PDF ────────────────────────────────────────────────
  dom.pdfUploadInput?.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      await uploadPdf(file);
      e.target.value = "";
    }
  });

  // ── Upload Image ──────────────────────────────────────────────
  // L'image est stockée temporairement et envoyée avec le prochain message.
  let pendingImageB64 = null;

  dom.imageUploadInput?.addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    if (file) {
      pendingImageB64 = await readImageAsBase64(file);
      showUploadStatus(`🖼️ Image prête : ${file.name}`, "info");
      e.target.value = "";
    }
  });

  // ── Gestionnaire d'envoi principal ────────────────────────────
  async function handleSend() {
    const text = dom.userInput.value.trim();
    if (!text || state.isStreaming) return;

    dom.userInput.value = "";
    if (dom.charCounter) dom.charCounter.textContent = "0 / 4000";

    const imageToSend = pendingImageB64;
    pendingImageB64 = null;
    if (imageToSend) {
      showUploadStatus("", "info");
      dom.uploadStatus.style.display = "none";
    }

    await sendMessageWithStream(text, imageToSend);
  }
}

// ── Lancement ─────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", init);
