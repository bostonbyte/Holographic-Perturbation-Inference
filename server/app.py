import sys
import os
import uuid
import struct
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add root to sys.path to allow importing hololib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hololib.compression import compress_matrix
from hololib.serializer import HoloSerializer

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Storage
MODEL_NAME = "gpt2"
device = "cpu"
if torch.backends.mps.is_available():
    device = "mps"
    
print(f"Loading {MODEL_NAME} on {device}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Ensure web/cache exists
os.makedirs("web/cache", exist_ok=True)

class PrefillRequest(BaseModel):
    text: str
    rank_fraction: float = 1.0 # Debug: Force full rank

class PrefillResponse(BaseModel):
    hologram_url: str
    num_tokens: int
    rank: int

@app.post("/prefill", response_model=PrefillResponse)
async def prefill(req: PrefillRequest):
    try:
        inputs = tokenizer(req.text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            past_key_values = outputs.past_key_values
            
        # past_key_values is Tuple[Tuple[K,V]] (Layers)
        # Each K/V is [Batch, NumHeads, SeqLen, HeadDim]
        
        u_list = []
        s_list = []
        v_list = []
        
        # Determine params from first layer
        # K has shape [B, H, S, D]
        k0 = past_key_values[0][0]
        num_layers = len(past_key_values)
        num_heads = k0.shape[1]
        seq_len = k0.shape[2]
        head_dim = k0.shape[3]
        
        # Calculate Rank (based on original length)
        rank = max(1, int(min(seq_len, head_dim) * req.rank_fraction))
        
        # Reduce Sequence Length (exclude last token) for Resonator Handoff
        # The client will re-process the last token to generate logits for the next step.
        hologram_seq_len = max(0, seq_len - 1)

        print(f"Compressing: L={num_layers} H={num_heads} S={seq_len}->{hologram_seq_len} D={head_dim} -> Target Rank {rank}")

        for layer_idx, (k, v) in enumerate(past_key_values):
            # ... (comments) ...

            # FIX: Slice to hologram_seq_len
            if hologram_seq_len < seq_len:
                k = k[:, :, :hologram_seq_len, :]
                v = v[:, :, :hologram_seq_len, :]

            # Check if empty (e.g. single token prompt)
            if k.shape[2] == 0:
                print("Warning: Cache is empty after slicing.")

            
            # Code:
            # compress_matrix returns u, s, v lists (per head).
            # We pass 'k' to it.
            # output is U, S, Vt of K.T?
            # In compression.py, I implemented SVD of K.t().
            # K.t() approx U S Vt.
            # K approx (U S Vt).T = V S Ut.
            # Q @ K.T = Q @ (U S Vt) = (Q @ U) * S @ Vt.
            # This matches Shader exactly!
            # Shader: Q @ U -> * S -> @ V.
            # So `bufferV` in Shader corresponds to `Vt` from SVD(K.t()).
            
            # Yes. So we save these.
            # Compress K (Key Cache)
            u_k, s_k, v_k_decomp = compress_matrix(k, req.rank_fraction)
            u_list.append(np.array(u_k))
            s_list.append(np.array(s_k))
            v_list.append(np.array(v_k_decomp))
            
            # Compress V (Value Cache)
            u_v, s_v, v_v_decomp = compress_matrix(v, req.rank_fraction)
            u_list.append(np.array(u_v))
            s_list.append(np.array(s_v))
            v_list.append(np.array(v_v_decomp))
            
        # Serialize
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.holo"
        filepath = os.path.join("web/cache", filename)
        
        # USE hologram_seq_len
        HoloSerializer.save(filepath, u_list, s_list, v_list, head_dim, rank, hologram_seq_len)
        
        return {
            "hologram_url": f"/cache/{filename}",
            "num_tokens": seq_len, # Return full count
            "rank": rank,
            "last_token_id": int(inputs.input_ids[0, -1])
        }
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Static Files
app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    # Listen on 8081 to avoid conflict with python http.server
    uvicorn.run(app, host="0.0.0.0", port=8081)
