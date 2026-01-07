import { HoloLoader } from './loader.js';
import { ResonatorEngine } from './engine_v2.js';

const engine = new ResonatorEngine();
let isEngineReady = false;
let vocab = {};

const txtInput = document.getElementById("txtInput");
const btnSend = document.getElementById("btnSend");
const history = document.getElementById("history");
const status = document.getElementById("status");

// Basic decoding (Byte Pair Encoding is hard in JS without library).
// We will use a simplified character map or just raw Token IDs if vocab fails.
async function loadVocab() {
    try {
        const r = await fetch('vocab.json');
        vocab = await r.json();
    } catch (e) {
        console.warn("No vocab.json found. Output will be IDs.");
    }
}

// Reverse dictionary for display
function decodeToken(id) {
    if (vocab[id]) return vocab[id].replace('Ġ', ' ');
    return `[${id}]`;
}

(async () => {
    try {
        status.innerText = "Initializing WebGPU...";
        await engine.init();

        status.innerText = "Downloading GPT-2 Weights (300MB)...";
        await engine.loadWeights('gpt2_weights.bin');

        await loadVocab();

        status.innerText = "System Ready. GPT-2 Loaded.";
        isEngineReady = true;
    } catch (e) {
        status.innerText = "Error: " + e.message;
        status.style.color = "red";
    }
})();

btnSend.onclick = async () => {
    if (!isEngineReady) return;
    const text = txtInput.value;
    if (!text) return;

    appendMessage("User", text);
    txtInput.value = "";

    status.innerText = "Server Processing...";

    try {
        // 1. Prefill
        const response = await fetch("/prefill", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: text })
        });
        const data = await response.json();

        status.innerText = `Receiving Hologram (Rank ${data.rank})...`;
        const holo = await HoloLoader.load(data.hologram_url);

        status.innerText = "Expanding Hologram on GPU...";
        await engine.loadHologram(holo);

        let currentToken = 50256; // Start token? Or use last token from server?
        // Server should return last token ID.
        // data.last_tokens?
        // Let's assume server logic returns `last_token_id`.
        // I need to update server/app.py to return this.
        // For now, assume 13 (period) or random?
        // Or re-tokenize 'text' locally?
        // Let's update server/app.py to return `last_token_id`.

        if (data.last_token_id) currentToken = data.last_token_id;

        status.innerText = "Dreaming...";
        const msgDiv = appendMessage("HPI", "");
        let genText = "";

        for (let i = 0; i < 20; i++) {
            const nextId = await engine.step(currentToken);
            const word = decodeToken(nextId);
            genText += word;
            msgDiv.innerText = genText;
            currentToken = nextId;

            // Allow UI update
            await new Promise(r => setTimeout(r, 0));
        }
        status.innerText = "Done.";

    } catch (e) {
        console.error(e);
        status.innerText = "Error: " + e.message;
    }
};

function appendMessage(role, text) {
    const div = document.createElement("div");
    div.className = "message " + role.toLowerCase();
    div.innerHTML = `<strong>${role}:</strong> ${text}`;
    history.appendChild(div);
    history.scrollTop = history.scrollHeight;
    return div;
}
