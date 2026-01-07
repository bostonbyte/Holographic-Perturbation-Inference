import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser(description="Holographic Perturbation Inference - Phase 1 Experiment")
    parser.add_argument("--model_name", type=str, default="gpt2", help="HuggingFace model name")
    parser.add_argument("--prompt", type=str, default="The quick brown fox jumps over the lazy dog. " * 50, help="Input prompt")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu, cuda, mps)")
    return parser.parse_args()

def compress_matrix(matrix, rank_fraction):
    """
    Compresses a matrix using SVD and truncates to the top 'rank_fraction' singular values.
    Returns the reconstructed matrix.
    """
    # Matrix shape: (Batch, Num_Heads, Seq_Len, Head_Dim)
    # We need to flatten to (Batch * Num_Heads * Seq_Len, Head_Dim) or similar for 2D SVD?
    # Actually, standard SVD is usually done per head or on the whole state. 
    # Let's do it per-head for correctness, as heads are independent.
    
    # But for simplicity and speed in this specific KV cache structure: 
    # The KV cache is usually [Batch, Num_Heads, Seq_Len, Head_Dim].
    # The "Holographic" theory implies compressing the TEMPORAL dimension (Seq_Len).
    # So we want to find a low-rank approximation of the Sequence history.
    
    # Reshape to 2D: [Batch * Num_Heads, Seq_Len, Head_Dim] -> treating Head_Dim as features and Seq_Len as samples?
    # No, we want to compress the stored history. 
    # Let's treat it as: For each head, we have a matrix of (Seq_Len, Head_Dim).
    # We want to approximate this matrix.
    
    b, h, s, d = matrix.shape
    compressed_matrix = torch.zeros_like(matrix)
    
    # We want to compress the stored history. 
    # Global SVD across all heads:
    # Reshape: [Batch, Num_Heads, Seq_Len, Head_Dim] -> [Batch, Seq_Len, Num_Heads * Head_Dim]
    matrix_flat = matrix.transpose(1, 2).reshape(b, s, h * d).float()
    
    # Apply SVD per batch item
    for i in range(b):
        u, s_val, v = torch.linalg.svd(matrix_flat[i], full_matrices=False)
        
        # Truncate
        k = max(1, int(min(s, h*d) * rank_fraction))
        
        # Reconstruct
        # U[:, :k] @ diag(S[:k]) @ V[:k, :]
        reconstructed = u[:, :k] @ torch.diag(s_val[:k]) @ v[:k, :]
        
        # Reconstruct: [k, Num_Heads*Head_Dim] -> [Seq_Len, Num_Heads*Head_Dim]
        reconstructed = u[:, :k] @ torch.diag(s_val[:k]) @ v[:k, :]
        
        # Reshape back to [Seq_Len, Num_Heads, Head_Dim]
        reconstructed_reshaped = reconstructed.view(s, h, d)
        
        # Transpose to [Num_Heads, Seq_Len, Head_Dim] to match target slice
        compressed_matrix[i] = reconstructed_reshaped.transpose(0, 1)

    return compressed_matrix.view(b, h, s, d).to(matrix.dtype)

def run_experiment():
    args = get_args()
    
    print(f"Loading model: {args.model_name}...")
    try:
        if args.device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif args.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    input_text = args.prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    print(f"Running baseline forward pass...")
    with torch.no_grad():
        # Get baseline logits and KV cache
        outputs = model(**inputs, use_cache=True)
        baseline_logits = outputs.logits
        past_key_values = outputs.past_key_values
    
    # past_key_values is a tuple of (key, value) for each layer
    # key/value shape: [Batch, Num_Heads, Seq_Len, Head_Dim]
    
    n_layers = len(past_key_values)
    print(f"Model has {n_layers} layers.")
    
    # Ranks to test (as fractions of full rank)
    rank_fractions = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0]
    drift_scores = []
    
    for rf in rank_fractions:
        print(f"Testing Rank Fraction: {rf}")
        
        new_past_key_values = []
        
        for layer_idx, (k, v) in enumerate(past_key_values):
            # Compress Key and Value
            k_approx = compress_matrix(k, rf)
            v_approx = compress_matrix(v, rf)
            new_past_key_values.append((k_approx, v_approx))
            
        # To test the result of the compression, we need to feed this PAST back into the model
        # and see if it predicts the NEXT token correctly.
        # However, standard forward() computes the KV cache for the GIVEN input.
        # To use our modified KV cache, we need to generate the NEXT token using the modified history.
        
        # We simulate the step: "The model has processed 'The quick brown fox'. Now predict the next token."
        # Normally we pass the whole sequence.
        # If we pass past_key_values, we just pass the last token.
        
        # Get the next token prediction using the modified cache
        last_token_logits = []
        
        # We need to run the model on the LAST token of the input, providing the cached history for the rest.
        # Input_ids for the next step is just the last token of the current sequence? Or do we imagine we are extending?
        # Let's try to reproduce the logits for the validation of the PROMPT itself, or the next token?
        # The outputs.logits contains logits for every position.
        # If we provide past_key_values, the model expects new input_ids to append.
        
        # Let's try this: 
        # 1. Use the compressed KV cache.
        # 2. Pass a DUMMY token (or the last token) and see if the output matches the original next-token prediction.
        
        # But wait, we want to know if the REPRESENTATION of the prompt is preserved.
        # The logits[0, -1, :] is the prediction for the word AFTER the prompt.
        # If we use the compressed KV cache, and pass a dummy input, do we get the same prediction?
        
        # Correct HuggingFace usage for past_key_values:
        # model(input_ids=next_token, past_key_values=compressed_cache)
        # But we don't know the next token. 
        # We just want to check if the state is consistent.
        
        # Actually, let's verify if the reconstructed KV cache works for generating the SAME next token prediction.
        # We need to supply the cache and NO input_ids? No, we must supply at least one token.
        # Usually, past_key_values represents tokens [0...N-1]. Input_ids is token [N].
        # Here, inputs has length N. outputs.past_key_values has length N.
        # This includes the computation of the last token.
        # So if we use this cache, we are ready to predict token N+1.
        
        # Let's ask the model to predict token N+1 using the full cache vs compressed cache.
        # To do this, we pass a dummy token " " (space) or similar, just to trigger a forward pass?
        # No, that adds a token.
        
        # What we really want to compare is:
        # A) Logits for token N+1 given full history.
        # B) Logits for token N+1 given COMPRESSED history.
        
        # The 'outputs.logits[0, -1, :]' IS the prediction for token N+1 based on the full sequence.
        # We want to reproduce THIS vector using the compressed cache.
        # BUT standard HF models don't easily let you "just compute logits from cache without new input".
        # The logits are the output of the LM Head, which takes the hidden state of the LAST token.
        # The hidden state of the last token depends on the attention over the Previous Keys/Values.
        
        # So, to test the compression, we should:
        # 1. Run inference on tokens [0...N-2]. Get Cache for N-1 tokens.
        # 2. Compress that Cache.
        # 3. Run inference on token [N-1] (the last token) using the Compressed Cache.
        # 4. Compare the resulting logits to the "True" logits from the full run.
        
        # Slice inputs
        input_ids = inputs.input_ids
        if input_ids.shape[1] < 2:
            print("Prompt too short for experiment.")
            return

        prefix_ids = input_ids[:, :-1]
        last_token_id = input_ids[:, -1:]
        
        # 1. Get Cache for Prefix
        with torch.no_grad():
            prefix_outputs = model(prefix_ids, use_cache=True)
            prefix_cache = prefix_outputs.past_key_values
            
        # 2. Compress Cache
        compressed_cache = []
        for l_k, l_v in prefix_cache:
            c_k = compress_matrix(l_k, rf)
            c_v = compress_matrix(l_v, rf)
            compressed_cache.append((c_k, c_v))
            
        # 3. Run inference on last token using Compressed Cache
        with torch.no_grad():
            # We must pass the compressed cache tuple
            # Note: HF expects tuples of tensors
            final_outputs = model(last_token_id, past_key_values=tuple(compressed_cache), use_cache=True)
            compressed_logits = final_outputs.logits[0, -1, :] # Logits for the next token
            
        # 4. Compare with Baseline
        # Baseline logits for the same position are outputs.logits[0, -1, :]
        true_logits = baseline_logits[0, -1, :]
        
        # Calc KL Divergence
        # Softmax both to get probabilities
        p_true = F.softmax(true_logits, dim=-1)
        p_approx = F.softmax(compressed_logits, dim=-1)
        
        # KL(P || Q) = sum(P * log(P/Q))
        kl_div = torch.sum(p_true * (torch.log(p_true + 1e-9) - torch.log(p_approx + 1e-9))).item()
        
        print(f"Rank: {rf} | KL Divergence: {kl_div:.6f}")
        drift_scores.append(kl_div)

    # Plot
    plt.figure()
    plt.plot(rank_fractions, drift_scores, marker='o')
    plt.title(f"Holographic Drill: SVD Compression vs Drift\nModel: {args.model_name}")
    plt.xlabel("Rank Fraction")
    plt.ylabel("KL Divergence (Lower is Better)")
    plt.grid(True)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/drift_curve.png")
    print("Plot saved to results/drift_curve.png")

if __name__ == "__main__":
    run_experiment()
