struct Params {
    batch_size: u32,
    in_dim: u32,
    out_dim: u32,
    seq_len: u32,
    layer_idx: u32,
    eps: f32, 
};

@group(0) @binding(0) var<uniform> params: Params;

// Generic Bindings (Read/Write to allow flexibility)
@group(0) @binding(1) var<storage, read_write> buffer1: array<f32>; // Input / QKV / CacheK
@group(0) @binding(2) var<storage, read_write> buffer2: array<f32>; // Weight / CacheV / CacheK_Read
@group(0) @binding(3) var<storage, read_write> buffer3: array<f32>; // Bias / CacheV_Read
@group(0) @binding(4) var<storage, read_write> buffer4: array<f32>; // Output
@group(0) @binding(5) var<storage, read_write> buffer5: array<f32>; // Spare / Residual

const PI: f32 = 3.14159265359;

// --- KERNEL: EMBEDDING ---
@compute @workgroup_size(64)
fn embedding(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dim_idx = global_id.x;
    if (dim_idx >= params.out_dim) { return; }
    
    // buffer1: [TokenID]
    // buffer2: WTE [Vocab, Dim]
    let token_id = u32(buffer1[0]);
    let w_idx = token_id * params.out_dim + dim_idx;
    
    // Dummy reads to force all bindings into layout
    let dummy3 = buffer3[0];
    let dummy5 = buffer5[0];
    
    buffer4[dim_idx] = buffer2[w_idx];
}

// --- KERNEL: LAYERNORM (Serial / Single Workgroup / Naive) ---
@compute @workgroup_size(1) 
fn layernorm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > 0u) { return; }
    
    // Dummy read to force binding 5 into layout
    let dummy5 = buffer5[0];
    
    var sum: f32 = 0.0;
    var sq_sum: f32 = 0.0;
    let N = params.in_dim;
    
    for (var i: u32 = 0u; i < N; i++) {
        let val = buffer1[i];
        sum += val;
        sq_sum += val * val;
    }
    
    let mean = sum / f32(N);
    let variance = max(0.0, (sq_sum / f32(N)) - (mean * mean));
    let inv_std = 1.0 / sqrt(variance + params.eps);
    
    for (var i: u32 = 0u; i < N; i++) {
        let gamma = buffer2[i];
        let beta = buffer3[i];
        buffer4[i] = ((buffer1[i] - mean) * inv_std) * gamma + beta;
    }
}

// --- KERNEL: MATMUL (Vector * Matrix + Bias) ---
// buffer1: Input [D_in]
// buffer2: Weight [D_out, D_in] (Row Major Layout [out, in])
// buffer3: Bias [D_out]
// buffer4: Output [D_out]
// Workgroup: 256 threads. Each computes one output element.
@compute @workgroup_size(256)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= params.out_dim) { return; }
    
    // Dummy read to force binding 5 into layout
    let dummy5 = buffer5[0];
    
    var dot: f32 = 0.0;
    let row_offset = out_idx * params.in_dim;
    
    // Row-Major layout: weight[out_idx, i] = weight[out_idx * in_dim + i]
    // Sequential access in inner loop (i moves by 1)
    for (var i: u32 = 0u; i < params.in_dim; i++) {
        dot += buffer1[i] * buffer2[row_offset + i];
    }
    
    buffer4[out_idx] = dot + buffer3[out_idx];
}

// --- KERNEL: GELU ---
@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.out_dim) { return; } // Use out_dim for target size
    let x = buffer1[i];
    let x3 = x * x * x;
    let inner = 0.79788456 * (x + 0.044715 * x3);
    
    // Dummy reads
    let dummy2 = buffer2[0];
    let dummy3 = buffer3[0];
    let dummy5 = buffer5[0];
    
    buffer4[i] = 0.5 * x * (1.0 + tanh(inner));
}

// --- KERNEL: ADD ---
@compute @workgroup_size(256)
fn add_residual(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.out_dim) { return; } // Use out_dim for target size
    
    // Dummy reads
    let dummy2 = buffer2[0];
    let dummy3 = buffer3[0];
    
    buffer4[i] = buffer1[i] + buffer5[i];
}

// --- KERNEL: ATTENTION HEAD ---
// buffer1: QKV Concatenated Input [3 * NumHeads * HeadDim]
// buffer2: Cache K [NumHeads * MaxSeq * HeadDim]
// buffer3: Cache V [NumHeads * MaxSeq * HeadDim]
// buffer4: Output [NumHeads * HeadDim]
// Workgroup: (64, 1, 1). Dispatched (12, 1, 1).
var<workgroup> wg_scores: array<f32, 1024>; // Shared memory for scores (Max stored context)

@compute @workgroup_size(64)
fn attention_head(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let head_idx = wg_id.x;     // 0..11 (one workgroup per head)
    let dim_idx = local_id.x;   // 0..63 (one thread per dimension)
    let tid = local_id.x;
    
    // Dummy read to force binding 5 into layout
    let dummy5 = buffer5[0];
    
    let head_dim = 64u;
    let num_heads = 12u;
    let max_seq = 1024u;
    let t = params.seq_len; // Current Token Index (0-based)
    
    let q_start = head_idx * head_dim;
    // Input is Q..K..V.. ?
    // Assuming GPT-2 HF structure: Conv1D output is [768*3].
    // chunk 0: Q (0..767). chunk 1: K. chunk 2: V.
    // Q for head i: i*64.
    
    let total_hidden = num_heads * head_dim;
    let my_q = buffer1[q_start + dim_idx];
    let my_k = buffer1[total_hidden + q_start + dim_idx];
    let my_v = buffer1[(2u * total_hidden) + q_start + dim_idx];
    
    // 1. Write to Cache (Global)
    // Cache layout: [Layer, Head, Time, Dim]
    let layer_size = num_heads * max_seq * head_dim;
    let head_size = max_seq * head_dim;
    
    let cache_layer_offset = params.layer_idx * layer_size;
    let cache_head_offset = head_idx * head_size;
    let cache_row_offset = t * head_dim; // t'th row
    
    let cache_base = cache_layer_offset + cache_head_offset;
    let cache_idx = cache_base + cache_row_offset + dim_idx;
    
    buffer2[cache_idx] = my_k;
    buffer3[cache_idx] = my_v;
    
    workgroupBarrier();
    
    // 2. Compute Scores
    // For each past timestep j <= t
    // Score[j] = dot(Q, K[j]) / sqrt(head_dim)
    
    for (var j: u32 = tid; j <= t; j += 64u) {
        var dot: f32 = 0.0;
        let k_row = cache_base + (j * head_dim);
        
        for (var d: u32 = 0u; d < 64u; d++) {
            let q_val = buffer1[q_start + d];
            let k_val = buffer2[k_row + d];
            dot += q_val * k_val;
        }
        wg_scores[j] = dot * 0.125; // 1/sqrt(64) = 0.125
    }
    workgroupBarrier();
    
    // 3. Softmax (Thread 0 Only)
    // Find Max
    if (tid == 0u) {
        var max_val: f32 = -1e20;
        for (var j: u32 = 0u; j <= t; j++) {
            if (wg_scores[j] > max_val) { max_val = wg_scores[j]; }
        }
        
        var sum_exp: f32 = 0.0;
        for (var j: u32 = 0u; j <= t; j++) {
            let e = exp(wg_scores[j] - max_val);
            wg_scores[j] = e;
            sum_exp += e;
        }
        
        for (var j: u32 = 0u; j <= t; j++) {
            wg_scores[j] /= sum_exp;
        }
    }
    workgroupBarrier();
    
    // 4. Weighted Sum (Parallel over dim_idx)
    var w_sum: f32 = 0.0;
    for (var j: u32 = 0u; j <= t; j++) {
        let sc = wg_scores[j];
        let v_val = buffer3[cache_base + (j * head_dim) + dim_idx];
        w_sum += sc * v_val;
    }
    
    buffer4[(head_idx * head_dim) + dim_idx] = w_sum;
}
