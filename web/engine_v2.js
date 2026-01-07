import { WeightsLoader } from './weights_v2.js';

export class ResonatorEngine {
    constructor() {
        this.device = null;
        this.weights = null;
        this.pipelines = {};
        this.paramsBuffer = null;

        // Model Config (GPT-2 Small)
        this.config = {
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            vocab_size: 50257
        };

        // Cache State
        this.cacheK = null;
        this.cacheV = null;
        this.seqLen = 0; // Current t

        // Intermediate Buffers
        this.bufs = {};
    }

    async init() {
        if (!navigator.gpu) throw new Error("WebGPU not supported");
        const adapter = await navigator.gpu.requestAdapter();

        // Request higher limits for large buffers (wte.weight is 154MB)
        this.device = await adapter.requestDevice({
            requiredLimits: {
                maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
            }
        });
        this.weights = new WeightsLoader(this.device);

        console.log("Compiling Shaders...");
        const shaderCode = await fetch('transformer.wgsl').then(r => r.text());
        const module = this.device.createShaderModule({ code: shaderCode });

        // Create Pipelines
        const createPipeline = (entry) => {
            return this.device.createComputePipeline({
                layout: 'auto',
                compute: { module, entryPoint: entry }
            });
        };

        this.pipelines.embedding = createPipeline('embedding');
        this.pipelines.layernorm = createPipeline('layernorm');
        this.pipelines.matmul = createPipeline('matmul'); // Generic matmul
        this.pipelines.gelu = createPipeline('gelu');
        this.pipelines.add = createPipeline('add_residual');
        this.pipelines.attn = createPipeline('attention_head');

        // Create Params Buffer
        this.paramsBuffer = this.device.createBuffer({
            size: 32, // 5 u32/f32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Create Intermediate Buffers
        const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
        const d = this.config.n_embd;

        this.bufs.x = this.createBuf(d * 4, usage);
        this.bufs.resid = this.createBuf(d * 4, usage);
        this.bufs.ln_out = this.createBuf(d * 4, usage);
        this.bufs.qkv = this.createBuf(3 * d * 4, usage);
        this.bufs.attn_out = this.createBuf(d * 4, usage);
        this.bufs.mlp_mid = this.createBuf(4 * d * 4, usage); // 3072
        this.bufs.mlp_gelu = this.createBuf(4 * d * 4, usage); // 3072 for gelu output
        this.bufs.mlp_out = this.createBuf(d * 4, usage);
        this.bufs.logits = this.createBuf(this.config.vocab_size * 4, usage);
        // Zero buffer for logits matmul (no bias needed for unembed)
        this.bufs.zero_bias = this.device.createBuffer({
            size: this.config.vocab_size * 4,
            usage: usage,
            mappedAtCreation: true
        });
        new Float32Array(this.bufs.zero_bias.getMappedRange()).fill(0);
        this.bufs.zero_bias.unmap();
        // We also need an input buffer for token_id
        this.bufs.input_ids = this.createBuf(4, usage);

        // KV Cache: [12, 12, 1024, 64]
        // Flattened: 12 * 12 * 1024 * 64 * 4 bytes = 144MB per cache.
        // Total 288MB.
        const cacheSize = 12 * 768 * 1024 * 4;
        this.cacheK = this.createBuf(cacheSize, usage);
        this.cacheV = this.createBuf(cacheSize, usage);

        // Create separate dummy buffers for each binding to prevent aliasing
        this.dummyBufs = [
            this.createBuf(4, usage), // dummy for binding 1
            this.createBuf(4, usage), // dummy for binding 2
            this.createBuf(4, usage), // dummy for binding 3
            this.createBuf(4, usage), // dummy for binding 4
            this.createBuf(4, usage), // dummy for binding 5
        ];

        console.log("Engine Initialized. Waiting for weights...");
    }

    createBuf(size, usage) {
        return this.device.createBuffer({ size, usage });
    }

    async loadWeights(url) {
        await this.weights.load(url);
        console.log("Weights Ready.");
    }

    async checkBufferNaN(name, buffer, size) {
        const readBuf = this.device.createBuffer({
            size: size * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        const cmd = this.device.createCommandEncoder();
        cmd.copyBufferToBuffer(buffer, 0, readBuf, 0, size * 4);
        this.device.queue.submit([cmd.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ);
        const data = new Float32Array(readBuf.getMappedRange());

        let nans = 0;
        let zeros = 0;
        let sum = 0;
        let sq_sum = 0;

        for (let i = 0; i < data.length; i++) {
            const v = data[i];
            if (isNaN(v)) nans++;
            if (v === 0) zeros++;
            sum += v;
            sq_sum += v * v;
        }

        const mean = sum / data.length;
        const variance = (sq_sum / data.length) - (mean * mean);
        const std = Math.sqrt(Math.max(0, variance));

        const msg = `[DEBUG] ${name}: Mean=${mean.toFixed(6)} Std=${std.toFixed(6)} Zeros=${zeros} NaNs=${nans} First5=[${data.slice(0, 5).join(', ')}]`;
        console.log(msg);

        const debugEl = document.getElementById("debug");
        if (debugEl) debugEl.innerText += msg + "\n";

        readBuf.unmap();
        readBuf.destroy();
    }

    // --- FORWARD PASS ---
    async forward(tokenId) {
        if (!this.weights.index) throw new Error("Weights not loaded");

        // After embedding phase, let's check:
        await this.checkBufferNaN('Initial Hidden (WTE+WPE)', this.bufs.x, 768);


        // 1. Upload Token as float (shader reads buffer as f32)
        const tokenInputArr = new Float32Array([tokenId]);
        this.device.queue.writeBuffer(this.bufs.input_ids, 0, tokenInputArr);

        // Diag Input
        if (this.seqLen < 5) { // Only log early steps
            await this.checkBufferNaN('Input IDs Buffer', this.bufs.input_ids, 1);
            // await this.checkBufferNaN('WTE Weight Slice', this.weights.getTensor('transformer.wte.weight'), 10);
        }

        // 2. Set Params (Common)
        this.updateParams(768, 768, 0);

        const cmd = this.device.createCommandEncoder();

        // --- Embedding ---
        // Input: input_ids. Weight: wte.weight. Output: x.
        // Also add Pos Embedding?
        // Simple GPT-2: x = wte[token] + wpe[t].
        // We run embedding twice? Or just sum.
        // Run Embedding WTE -> x.
        // Run Embedding WPE (input=t) -> resid.
        // Add x+resid -> x.

        // Dispatch Embedding WTE
        this.dispatch(cmd, this.pipelines.embedding, [
            this.bufs.input_ids,
            this.weights.getTensor('transformer.wte.weight'),
            null,
            this.bufs.x
        ], [12, 1, 1]); // 768 / 64 = 12 workgroups

        // Dispatch Embedding WPE
        // We need a buffer for 't'. Reuse input_ids? No it has token_id.
        // Create temp buf for t? 
        // Just upload t to 'input_ids' again? Input IDs is used by WTE.
        // We can overwrite it if we wait? No, queue write is unordered wrt cmd?
        // Queue writes happen before Submits.
        // We can't update buffer mid-command-buffer without splitting submit.
        // Solution: Use a second input slot (params.seq_len)? 
        // Or just map wpe[t] on CPU and upload? 
        // wpe[t] is size 768. Uploading 3KB is fast.
        // Let's do that for simplicity. 
        // But 'wpe.weight' is on GPU.
        // Okay, we need a separate buffer for 't'.
        // Let's create 'pos_id_buf' and write 't' to it.
        // In this 'forward' call, we prefer submitting ONE BIG BATCH.
        // So we need 'input_ids' and 'pos_ids' buffers.
        // Add 'this.bufs.pos_ids' to init. (I'll skip logic to keep code shorter, assume t=0 for MVP? NO).
        // Let's create `pos_ids`.
        const pos_ids = this.createBuf(4, GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE);
        this.device.queue.writeBuffer(pos_ids, 0, new Float32Array([this.seqLen]));

        // Output WPE to 'resid'.
        this.dispatch(cmd, this.pipelines.embedding, [
            pos_ids,
            this.weights.getTensor('transformer.wpe.weight'),
            null,
            this.bufs.resid
        ], [12, 1, 1]);

        // Add WTE + WPE -> ln_out (temp), then copy to x
        this.dispatch(cmd, this.pipelines.add, [
            this.bufs.x, // in (x)
            null, null,
            this.bufs.ln_out, // out (temp)
            this.bufs.resid // other
        ], [3, 1, 1]); // 768 / 256 = 3 WGs

        // Copy result back to x
        cmd.copyBufferToBuffer(this.bufs.ln_out, 0, this.bufs.x, 0, 768 * 4);

        // Submit embedding phase
        this.device.queue.submit([cmd.finish()]);

        // DIAGNOSTIC
        // await this.checkBufferNaN('Initial Hidden (WTE+WPE)', this.bufs.x, 768);

        // --- LAYERS 0..11 ---
        for (let i = 0; i < 12; i++) {
            const prefix = `transformer.h.${i}.`;

            // Create fresh encoder for this layer
            let loopCmd = this.device.createCommandEncoder();

            // Save x (hidden) to resid for residual connection
            loopCmd.copyBufferToBuffer(this.bufs.x, 0, this.bufs.resid, 0, 768 * 4);

            // 1. LN1
            this.updateParams(768, 768, i); // Ensure correct layerIdx
            this.dispatch(loopCmd, this.pipelines.layernorm, [
                this.bufs.x,
                this.weights.getTensor(prefix + 'ln_1.weight'),
                this.weights.getTensor(prefix + 'ln_1.bias'),
                this.bufs.ln_out
            ], [1, 1, 1]); // Serial LN

            // Submit LN1 before changing params
            this.device.queue.submit([loopCmd.finish()]);
            await this.checkBufferNaN(`Layer ${i} LN1`, this.bufs.ln_out, 768);

            // 2. QKV Matmul
            this.updateParams(2304, 768, i);
            loopCmd = this.device.createCommandEncoder();

            this.dispatch(loopCmd, this.pipelines.matmul, [
                this.bufs.ln_out,
                this.weights.getTensor(prefix + 'attn.c_attn.weight'),
                this.weights.getTensor(prefix + 'attn.c_attn.bias'),
                this.bufs.qkv
            ], [9, 1, 1]); // 2304 / 256 = 9

            this.device.queue.submit([loopCmd.finish()]);
            await this.checkBufferNaN(`Layer ${i} QKV`, this.bufs.qkv, 2304);

            loopCmd = this.device.createCommandEncoder();

            // 3. Attention Kernel
            // (BindGroup already uses paramsBuffer which we just updated)
            // We bind the FULL cache buffers because the shader handles the layer offset.
            const bindGroupAttn = this.device.createBindGroup({
                layout: this.pipelines.attn.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: this.paramsBuffer } },
                    { binding: 1, resource: { buffer: this.bufs.qkv } },
                    { binding: 2, resource: { buffer: this.cacheK } },
                    { binding: 3, resource: { buffer: this.cacheV } },
                    { binding: 4, resource: { buffer: this.bufs.attn_out } },
                    { binding: 5, resource: { buffer: this.dummyBufs[4] } }
                ]
            });

            const passAttn = loopCmd.beginComputePass();
            passAttn.setPipeline(this.pipelines.attn);
            passAttn.setBindGroup(0, bindGroupAttn);
            passAttn.dispatchWorkgroups(12, 1, 1);
            passAttn.end();

            this.device.queue.submit([loopCmd.finish()]);
            await this.checkBufferNaN(`Layer ${i} AttnOut`, this.bufs.attn_out, 768);

            loopCmd = this.device.createCommandEncoder();

            // 4. Projection
            this.updateParams(768, 768, i);
            this.dispatch(loopCmd, this.pipelines.matmul, [
                this.bufs.attn_out,
                this.weights.getTensor(prefix + 'attn.c_proj.weight'),
                this.weights.getTensor(prefix + 'attn.c_proj.bias'),
                this.bufs.ln_out // Reuse buffer
            ], [3, 1, 1]);

            // 5. Residual Add
            this.dispatch(loopCmd, this.pipelines.add, [
                this.bufs.ln_out, // proj_out
                null, null,
                this.bufs.x,      // x outputs here
                this.bufs.resid   // old x
            ], [3, 1, 1]);

            this.device.queue.submit([loopCmd.finish()]);
            await this.checkBufferNaN(`Layer ${i} After Attn`, this.bufs.x, 768);
            loopCmd = this.device.createCommandEncoder();

            // 6. LN2
            // copy x to resid
            loopCmd.copyBufferToBuffer(this.bufs.x, 0, this.bufs.resid, 0, 768 * 4);

            this.dispatch(loopCmd, this.pipelines.layernorm, [
                this.bufs.x,
                this.weights.getTensor(prefix + 'ln_2.weight'),
                this.weights.getTensor(prefix + 'ln_2.bias'),
                this.bufs.ln_out
            ], [1, 1, 1]);

            // 7. MLP c_fc (768 -> 3072)
            this.device.queue.submit([loopCmd.finish()]);
            await this.checkBufferNaN(`Layer ${i} LN2`, this.bufs.ln_out, 768);
            this.updateParams(3072, 768, i);
            loopCmd = this.device.createCommandEncoder();

            this.dispatch(loopCmd, this.pipelines.matmul, [
                this.bufs.ln_out,
                this.weights.getTensor(prefix + 'mlp.c_fc.weight'),
                this.weights.getTensor(prefix + 'mlp.c_fc.bias'),
                this.bufs.mlp_mid
            ], [12, 1, 1]); // 3072/256 = 12

            // 8. GeLU
            this.dispatch(loopCmd, this.pipelines.gelu, [
                this.bufs.mlp_mid,
                null, null,
                this.bufs.mlp_gelu
            ], [12, 1, 1]);

            // 9. MLP c_proj (3072 -> 768)
            this.device.queue.submit([loopCmd.finish()]);
            await this.checkBufferNaN(`Layer ${i} GeLU`, this.bufs.mlp_gelu, 3072);
            this.updateParams(768, 3072, i);
            loopCmd = this.device.createCommandEncoder();

            this.dispatch(loopCmd, this.pipelines.matmul, [
                this.bufs.mlp_gelu,
                this.weights.getTensor(prefix + 'mlp.c_proj.weight'),
                this.weights.getTensor(prefix + 'mlp.c_proj.bias'),
                this.bufs.mlp_out
            ], [3, 1, 1]);

            // 10. Residual Add
            this.dispatch(loopCmd, this.pipelines.add, [
                this.bufs.mlp_out,
                null, null,
                this.bufs.x,
                this.bufs.resid
            ], [3, 1, 1]);

            this.device.queue.submit([loopCmd.finish()]);
            await this.checkBufferNaN(`Layer ${i} Final x`, this.bufs.x, 768);
        } // End Layer Loop

        // Final LN
        const finalCmd = this.device.createCommandEncoder();
        this.dispatch(finalCmd, this.pipelines.layernorm, [
            this.bufs.x,
            this.weights.getTensor('transformer.ln_f.weight'),
            this.weights.getTensor('transformer.ln_f.bias'),
            this.bufs.ln_out // Reuse
        ], [1, 1, 1]);

        // Unembed (Logits)
        // Weight: wte.weight (Transposed? No, tied weights).
        // wte is [Vocab, Dim]. Input [Dim].
        // Matmul wants [Out, In]. so [Vocab, Dim].
        // This matches wte layout (Vocab rows).
        this.device.queue.submit([finalCmd.finish()]);
        this.updateParams(this.config.vocab_size, 768, 0);
        const logitCmd = this.device.createCommandEncoder();

        this.dispatch(logitCmd, this.pipelines.matmul, [
            this.bufs.ln_out,
            this.weights.getTensor('transformer.wte.weight'),
            this.bufs.zero_bias, // Zero bias for unembed
            this.bufs.logits
        ], [197, 1, 1]); // 50257 / 256 ~ 197

        // Readback Logits (Top 1)
        // Copy to readback buffer?
        // We'll just read mapped buffer for now (slow sync).
        // Ideally run ArgMax on GPU. 
        // For MVP, read back 50257 floats? (200KB). Fast enough.

        this.device.queue.submit([logitCmd.finish()]);

        // ... Reading logic in `step` wrapper ...
    }

    updateParams(outDim, inDim = 768, layerIdx = 0) {
        const paramsData = new ArrayBuffer(32);
        const view = new DataView(paramsData);
        view.setUint32(0, 1, true);   // batch
        view.setUint32(4, inDim, true);
        view.setUint32(8, outDim, true);
        view.setUint32(12, this.seqLen, true);
        view.setUint32(16, layerIdx, true); // layer_idx
        view.setFloat32(20, 1e-5, true); // eps
        this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);
    }

    dispatch(cmd, pipeline, buffers, size) {
        // Track which bindings are used
        const usedBindings = new Set([0]);
        const entries = [
            { binding: 0, resource: { buffer: this.paramsBuffer } }
        ];

        // Add provided buffers
        buffers.forEach((b, i) => {
            const bindingIdx = i + 1;
            if (b) {
                entries.push({ binding: bindingIdx, resource: { buffer: b } });
                usedBindings.add(bindingIdx);
            }
        });

        // Fill missing bindings 1-5 with separate dummy buffers (to prevent aliasing)
        for (let i = 1; i <= 5; i++) {
            if (!usedBindings.has(i)) {
                entries.push({ binding: i, resource: { buffer: this.dummyBufs[i - 1] } });
            }
        }

        const bg = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries
        });

        const pass = cmd.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(...size);
        pass.end();
    }

    // Simple greedy step
    async step(tokenId) {
        await this.forward(tokenId);

        // Read logits
        // Need to copy to mappable buffer
        const readBuf = this.device.createBuffer({
            size: this.config.vocab_size * 4,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const cmd = this.device.createCommandEncoder();
        cmd.copyBufferToBuffer(this.bufs.logits, 0, readBuf, 0, this.config.vocab_size * 4);
        this.device.queue.submit([cmd.finish()]);

        await readBuf.mapAsync(GPUMapMode.READ);
        const logits = new Float32Array(readBuf.getMappedRange());

        // Debug: Log first few logits and check for issues
        console.log("First 10 logits:", Array.from(logits.slice(0, 10)));
        console.log("logits[0]:", logits[0], "logits[1]:", logits[1], "logits[100]:", logits[100]);

        // Check for NaN
        let nanCount = 0;
        let zeroCount = 0;
        for (let i = 0; i < Math.min(1000, logits.length); i++) {
            if (isNaN(logits[i])) nanCount++;
            if (logits[i] === 0) zeroCount++;
        }
        console.log(`First 1000: ${nanCount} NaN, ${zeroCount} zeros`);

        // Argmax
        let maxVal = -Infinity;
        let maxIdx = 0;
        for (let i = 0; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxIdx = i;
            }
        }

        console.log(`argmax: idx=${maxIdx}, val=${maxVal}`);

        readBuf.unmap();
        this.seqLen++;
        return maxIdx;
    }

    async loadHologram(holo) {
        console.log("Expanding Hologram to Full Cache...");
        // holo.layers[l][h] has {u, s, v}
        // CacheK layout: [12, 12, 1024, 64] flat floats
        // We need to fill indices 0..seqLen-1.

        // Host buffers
        const cacheK_CPU = new Float32Array(12 * 12 * 1024 * 64);
        const cacheV_CPU = new Float32Array(12 * 12 * 1024 * 64);

        const rank = holo.rank;
        const seq = holo.seqLen;
        const dim = holo.headDim;
        this.seqLen = seq;

        for (let l = 0; l < 12; l++) {
            // Layer K is at index 2*l, Layer V is at 2*l+1
            // (Assuming server serialized: Layer0_K, Layer0_V, Layer1_K, ...)
            const layerDataK = holo.layers[l * 2];
            const layerDataV = holo.layers[l * 2 + 1];

            for (let h = 0; h < 12; h++) {
                // --- K Reconstruction ---
                {
                    const U = layerDataK[h].u;
                    const S = layerDataK[h].s;
                    const V = layerDataK[h].v;

                    // Offset in cache
                    const headOffset = (l * 12 * 1024 * 64) + (h * 1024 * 64);

                    for (let t = 0; t < seq; t++) {
                        const rowOffset = headOffset + (t * 64);
                        for (let d = 0; d < 64; d++) {
                            let val = 0;
                            for (let r = 0; r < rank; r++) {
                                // Reconstruct K[t, d]
                                const u_val = U[d * rank + r];
                                const s_val = S[r];
                                const v_val = V[r * seq + t];
                                val += u_val * s_val * v_val;
                            }
                            cacheK_CPU[rowOffset + d] = val;
                        }
                    }
                }

                // --- V Reconstruction ---
                {
                    const U = layerDataV[h].u;
                    const S = layerDataV[h].s;
                    const V = layerDataV[h].v;
                    const headOffset = (l * 12 * 1024 * 64) + (h * 1024 * 64);

                    for (let t = 0; t < seq; t++) {
                        const rowOffset = headOffset + (t * 64);
                        for (let d = 0; d < 64; d++) {
                            let val = 0;
                            for (let r = 0; r < rank; r++) {
                                // Reconstruct V[t, d]
                                const u_val = U[d * rank + r];
                                const s_val = S[r];
                                const v_val = V[r * seq + t];
                                val += u_val * s_val * v_val;
                            }
                            cacheV_CPU[rowOffset + d] = val;
                        }
                    }
                }
            }
        }

        // Upload
        this.device.queue.writeBuffer(this.cacheK, 0, cacheK_CPU);
        this.device.queue.writeBuffer(this.cacheV, 0, cacheV_CPU);
        console.log("Hologram Expanded and Uploaded.");

        // Verify Cache K (First Head, First Token)
        // Offset 0.
        await this.checkBufferNaN('CacheK Head0 Token0', this.cacheK, 64);
    }
}
