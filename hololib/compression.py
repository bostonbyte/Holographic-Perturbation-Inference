import torch

def compress_matrix(matrix, rank_fraction):
    """
    Compresses a matrix using SVD and truncates to the top 'rank_fraction' singular values.
    Returns: U, S, V (The decomposed components, NOT the reconstructed matrix)
    """
    # matrix shape: [Batch, Num_Heads, Seq_Len, Head_Dim]
    b, h, s, d = matrix.shape
    
    # We want to compress the stored history. 
    # Global SVD across all heads:
    # Reshape: [Batch, Num_Heads, Seq_Len, Head_Dim] -> [Batch, Seq_Len, Num_Heads * Head_Dim]
    
    # NOTE: The server might pass a single batch item (b=1).
    
    # For storage, we want [NumHeads, HeadDim, Rank] format for U?
    # Or do we stick to the "Global SVD" logic from Experiment 1?
    # In Exp 1 we did: matrix_flat = matrix.transpose(1, 2).reshape(b, s, h * d)
    # Then SVD on (s, h*d).
    # Then Reconstructed.
    
    # But for the Protocol and Resonator, the Resonator Shader expects:
    # U: [D, K] per head?
    # My shader `resonator.wgsl` has `bufferU : array<f32>; // Shape [D, K]`.
    # And it loops `for (var d ...)` and `let u_val = bufferU[d * params.K + k_idx]`.
    # This implies the shader handles ONE head.
    # So we should perform SVD **Per Head**.
    
    # Experiment 1 logic used "Global SVD" which mixes heads. 
    # That was acceptable for demonstrating "Drift" (compression exists).
    # But for the "Resonator" implementation in WGSL, I wrote it to process "InputQ" (Size D).
    # This implies the shader runs Per Head.
    
    # So `compress_matrix` here should operate Per Head.
    
    # Input: [Batch, Num_Heads, Seq_Len, Head_Dim]
    # We ignore Batch for now (assume 1).
    
    u_list = []
    s_list = []
    v_list = []
    
    # Targets
    target_rank = max(1, int(min(s, d) * rank_fraction)) # Local rank usually bounded by d (64)
    # Be careful: In Exp 1 we found Short Sequence -> Rank 1.
    # If we do Per-Head SVD, we are limited by 'd' (64).
    # Compressing 64 -> 32 is only 50%.
    # The "Holographic" win comes from compressing SeqLen (4096) -> Rank (32).
    # So we need SVD of (SeqLen, HeadDim).
    # Matrix M is (SeqLen, HeadDim).
    # U (SeqLen, Rank), S (Rank), Vt (Rank, HeadDim).
    # We store U, S, V.
    
    # Wait, the Shader:
    # Input Q (HeadDim).
    # We want Attention Scores (SeqLen).
    # Standard: Scores = Q @ K.T   (1, D) @ (D, SeqLen) -> (1, SeqLen).
    
    # SVD of K.T (which is D x SeqLen)?
    # K.T = U * S * Vt. 
    # U is (D, Rank). S is (Rank). Vt is (Rank, SeqLen).
    # Scores = Q @ (U S Vt)
    # Scores = (Q @ U) * S @ Vt.
    # Dimensions: (1, D) @ (D, R) -> (1, R).
    # (1, R) * R -> (1, R).
    # (1, R) @ (R, SeqLen) -> (1, SeqLen).
    # This matches the Shader logic!
    
    # So we need to SVD **K.transposed** (D, SeqLen).
    # Or SVD of K (SeqLen, D) -> U (SeqLen, R), S, Vt (R, D).
    # Then K.T = V * S * Ut.
    # Let's just SVD(K.T).
    
    # Iterate heads
    for head_idx in range(h):
        # K_head: [Seq_Len, Head_Dim]
        k_head = matrix[0, head_idx, :, :].float()
        
        # We want K_T: [Head_Dim, Seq_Len]
        k_t = k_head.t() # (D, S)
        
        # SVD
        # U (D, D) or (D, Min), S (Min), V (Min, S)
        # We want rank R
        try:
             # Move to CPU for stable SVD
            k_t_cpu = k_t.cpu()
            
            U, S, V = torch.linalg.svd(k_t_cpu, full_matrices=False)
            
            # DEBUG
            if head_idx == 0:
                print(f"HEAD 0 SVD: K_T shape {k_t_cpu.shape}")
                print(f"  S val top 3: {S[:3]}")
                print(f"  U mean: {U.mean()} V mean: {V.mean()}")

        except Exception as e:
             print(f"SVD Failed: {e}")
             U = torch.zeros(d, target_rank)
             S = torch.zeros(target_rank)
             V = torch.zeros(target_rank, s)
        
        # Truncate
        eff_rank = max(1, int(min(d, s) * rank_fraction))
        
        u_trunc = U[:, :eff_rank]
        s_trunc = S[:eff_rank]
        v_trunc = V[:eff_rank, :]
        
        u_list.append(u_trunc.detach().cpu().numpy())
        s_list.append(s_trunc.detach().cpu().numpy())
        v_list.append(v_trunc.detach().cpu().numpy())
        
    return u_list, s_list, v_list
