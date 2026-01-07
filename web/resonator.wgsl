struct Params {
    D : u32,
    K : u32,
    T : u32,
}

@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read> bufferU : array<f32>; // Shape [D, K] (Row-major: D rows, K cols) or Column-major? Let's assume Row Major U[d*K + k]
@group(0) @binding(2) var<storage, read> bufferS : array<f32>; // Shape [K]
@group(0) @binding(3) var<storage, read> bufferV : array<f32>; // Shape [K, T] -> Let's assume V is V^T in memory? 
// If V is (K, T), we access V[k*T + t].

@group(0) @binding(4) var<storage, read> inputQ : array<f32>; // Shape [D]
@group(0) @binding(5) var<storage, read_write> outputScores : array<f32>; // Shape [T]

// Shared memory for intermediate vector (Q @ U * S) of size K
// Max K usually 256. Workgroup size 256 fits.
var<workgroup> projection : array<f32, 256>; 

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>) {
    let t = global_id.x; // Each thread handles one output time step? No, T can be large (4096).
    // If T > 256, we need multiple workgroups.
    
    let k_idx = local_id.x; // 0..255
    
    // Step 1: Project Q into Hologram Space (Calculate Q @ U)
    // This produces a vector of size K.
    // Each thread 'k' can compute one element of the result vector?
    // Element k = DotProduct(Q, U[:, k]).
    // Q is size D. U[:, k] is the k-th column of U.
    // This requires looping over D.
    
    if (k_idx < params.K) {
        var sum : f32 = 0.0;
        for (var d = 0u; d < params.D; d = d + 1u) {
            let u_val = bufferU[d * params.K + k_idx];
            let q_val = inputQ[d];
            sum = sum + u_val * q_val;
        }
        
        // Step 2: Scale by Singular Value S
        let s_val = bufferS[k_idx];
        projection[k_idx] = sum * s_val;
    }
    
    workgroupBarrier();
    
    // Step 3: Expand back to Time dimension (Project @ V)
    // Result[t] = DotProduct(Projected, V[:, t])
    // V[:, t] is the t-th column of V (size K).
    // So we iterate over K.
    
    // We want to parallelize over T (Output Size).
    // The current dispatch launches threads over dim T?
    // If we have dispatch(ceil(T/256)), then global_id.x is 't'.
    
    // But Step 1 (Q@U) assumes local_id covers 'K'.
    // If K and BlockSize are both 256, this matches.
    // BUT we are doing this redundantly for EVERY Workgroup (Chunk of T).
    // That's actually fine. D and K are small. T is huge.
    // Recomputing the projection vector 16 times (for T=4096 / 256 = 16 blocks) is negligible cost compared to memory traffic.
    
    if (t < params.T) {
        var score : f32 = 0.0;
        for (var k = 0u; k < params.K; k = k + 1u) {
            let v_val = bufferV[k * params.T + t]; // V stored as [K, T]
            score = score + projection[k] * v_val;
        }
        outputScores[t] = score;
    }
}
