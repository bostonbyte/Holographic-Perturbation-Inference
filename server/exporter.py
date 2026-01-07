import torch
from transformers import GPT2LMHeadModel
import json
import struct
import os
import numpy as np

def export_gpt2():
    model_name = "gpt2"
    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    
    # We strip the 'transformer.' prefix and handle some specific mappings if needed.
    # HF GPT-2 keys:
    # wte.weight, wpe.weight
    # h.0.ln_1.weight, h.0.ln_1.bias
    # h.0.attn.c_attn.weight, h.0.attn.c_attn.bias
    # ...
    # ln_f.weight, ln_f.bias
    
    state_dict = model.state_dict()
    
    tensors = {}
    binary_data = bytearray()
    current_offset = 0
    
    print("Exporting weights...")
    
    for key, tensor in state_dict.items():
        # Clean naming? Keep original for simplicity.
        # Ensure float16
        # Note: HF GPT-2 weights are usually Float32.
        # c_attn weights are [768, 2304]. WebGPU matmul (vec * mat) works fine with this.
        # Linear weights (c_proj) are [768, 768].
        
        # Standardize to [out_dim, in_dim]
        # HF GPT-2 Conv1D (c_attn, c_fc, c_proj) are [in, out]
        # HF GPT-2 Linear/Embedding (wte, wpe, ln) are [out, in] or [dim]
        
        data_np = tensor.cpu().detach().numpy().astype(np.float16)
        
        if len(data_np.shape) == 2:
            # If it's a Conv1D layer, transpose it to [out, in]
            if ".attn.c_attn" in key or ".attn.c_proj" in key or ".mlp.c_fc" in key or ".mlp.c_proj" in key:
                print(f"  Transposing Conv1D {key} {data_np.shape} -> {data_np.T.shape}")
                data_np = data_np.T
            else:
                print(f"  Keeping [out, in] {key} {data_np.shape}")
        
        # NaN check
        if np.isnan(data_np).any() or np.isinf(data_np).any():
            print(f"  WARNING: Tensor {key} contains NaN/Inf values!")
        
        data_bytes = data_np.tobytes()
        size_bytes = len(data_bytes)
        
        tensors[key] = {
            "offset": current_offset,
            "size": size_bytes,
            "shape": list(data_np.shape),
            "dtype": "float16"
        }
        
        binary_data.extend(data_bytes)
        current_offset += size_bytes
        
    # Create Header
    header_json = json.dumps(tensors)
    header_bytes = header_json.encode('utf-8')
    header_len = len(header_bytes)
    
    output_path = "web/gpt2_weights.bin"
    os.makedirs("web", exist_ok=True)
    
    with open(output_path, "wb") as f:
        # Write 4-byte header length
        f.write(struct.pack('<I', header_len))
        # Write Header
        f.write(header_bytes)
        # Write Payload
        f.write(binary_data)
        
    print(f"Exported {current_offset / 1024 / 1024:.2f} MB to {output_path}")
    print(f"Header Size: {header_len} bytes")

if __name__ == "__main__":
    export_gpt2()
