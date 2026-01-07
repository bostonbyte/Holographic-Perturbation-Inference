async function init() {
    if (!navigator.gpu) {
        log("WebGPU not supported on this browser.");
        return;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        log("No WebGPU adapter found.");
        return;
    }
    const device = await adapter.requestDevice();
    log(`WebGPU Device: ${adapter.limits.maxComputeWorkgroupSizeX} max workgroup X`);

    // --- Configuration ---
    const D = 128;   // Head Dimension
    const K = 32;    // Hologram Rank
    const T = 4096;  // Sequence Length (Context)

    // Model Params (Llama-3-8B equivalent)
    // 32 Layers, 32 Heads.
    // Total Ops per Token for Attention: 
    // Naive: 2 * D * T per head. Total = 32 * 2 * 128 * 4096 = 33.5 MFLOPs.
    // Holographic: 2 * (D*K + K + K*T) = 2 * (4k + 32 + 131k) = 270 KFLOPs per head.
    // Total = 32 * 270k = 8.6 MFLOPs.
    // Speedup Factor: ~4x? No, 131k vs 524k is 4x.

    log(`Config: D=${D}, K=${K}, T=${T}`);
    log(`Theoretical Compression: ${(K * T + K * D + K) / (D * T) * 100}% of memory`);

    // --- Data Gen ---
    const dataU = new Float32Array(D * K).map(() => Math.random());
    const dataS = new Float32Array(K).map(() => Math.random());
    const dataV = new Float32Array(K * T).map(() => Math.random());
    const inputQ = new Float32Array(D).map(() => Math.random());

    // --- Buffers ---
    const bufferU = createBuffer(device, dataU, GPUBufferUsage.STORAGE);
    const bufferS = createBuffer(device, dataS, GPUBufferUsage.STORAGE);
    const bufferV = createBuffer(device, dataV, GPUBufferUsage.STORAGE);
    const bufferQ = createBuffer(device, inputQ, GPUBufferUsage.STORAGE);

    const resultSize = T * 4;
    const outputBuffer = device.createBuffer({
        size: resultSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });
    const readbackBuffer = device.createBuffer({
        size: resultSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const paramsValues = new Uint32Array([D, K, T, 0]); // Pad to 16 bytes alignment
    const paramsBuffer = createBuffer(device, paramsValues, GPUBufferUsage.UNIFORM);

    // --- Shader ---
    const shaderCode = await fetch('resonator.wgsl').then(r => r.text());
    const shaderModule = device.createShaderModule({ code: shaderCode });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ]
    });

    const pipeline = device.createComputePipeline({
        layout: "auto", // device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: "main" }
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: bufferU } },
            { binding: 2, resource: { buffer: bufferS } },
            { binding: 3, resource: { buffer: bufferV } },
            { binding: 4, resource: { buffer: bufferQ } },
            { binding: 5, resource: { buffer: outputBuffer } },
        ]
    });

    // --- CPU Verification ---
    log("Verifying correctness (CPU)...");
    const cpuProjected = new Float32Array(K);
    // Q @ U * S
    for (let k = 0; k < K; k++) {
        let sum = 0;
        for (let d = 0; d < D; d++) sum += inputQ[d] * dataU[d * K + k];
        cpuProjected[k] = sum * dataS[k];
    }
    const cpuOutput = new Float32Array(T);
    // Proj @ V
    for (let t = 0; t < T; t++) {
        let sum = 0;
        for (let k = 0; k < K; k++) sum += cpuProjected[k] * dataV[k * T + t];
        cpuOutput[t] = sum;
    }

    // Run 1 pass for check
    const commandEncoder = device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    // Dispatch: Threads = T. Workgroup = 256.
    pass.dispatchWorkgroups(Math.ceil(T / 256));
    pass.end();
    commandEncoder.copyBufferToBuffer(outputBuffer, 0, readbackBuffer, 0, resultSize);
    device.queue.submit([commandEncoder.finish()]);

    await readbackBuffer.mapAsync(GPUMapMode.READ);
    const gpuOutput = new Float32Array(readbackBuffer.getMappedRange());

    // Diff
    let maxDiff = 0;
    for (let i = 0; i < T; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(gpuOutput[i] - cpuOutput[i]));
    }
    readbackBuffer.unmap();
    log(`Correctness Check: Max Diff = ${maxDiff.toFixed(6)}`);
    if (maxDiff > 0.1) {
        log("ERROR: GPU computation incorrect.");
        return;
    } else {
        log("SUCCESS: GPU matches CPU.");
    }

    // --- Benchmark Loop ---
    document.getElementById("btnRun").onclick = async () => {
        log("Running Speed Test (1000 Iterations)...");
        const ITER = 1000;
        const start = performance.now();

        const encoder = device.createCommandEncoder();
        const bPass = encoder.beginComputePass();
        bPass.setPipeline(pipeline);
        bPass.setBindGroup(0, bindGroup);
        for (let i = 0; i < ITER; i++) {
            bPass.dispatchWorkgroups(Math.ceil(T / 256));
        }
        bPass.end();
        device.queue.submit([encoder.finish()]);

        // Wait for completion (cheap way: map buffer again)
        await device.queue.onSubmittedWorkDone();

        const end = performance.now();
        const dt = end - start;
        const tps = (ITER / dt) * 1000;

        log(`Time: ${dt.toFixed(2)}ms`);
        log(`Kernel Speed: ${tps.toFixed(0)} passes/sec`);

        // Assume 32 heads * 32 layers = 1024 such passes per "Token Generation"
        // Wait, one pass is ONE head. 
        // A model has ~1000 head-passes per token.
        // So Real TPS = Kernel Speed / 1000.
        // If Kernel Speed is 50,000, then Real TPS = 50.

        log(`Estimated Token Gen Speed (Llama-3-8B): ${(tps / (32 * 32)).toFixed(1)} tokens/sec`);
    };
}

function createBuffer(device, data, usage) {
    const buffer = device.createBuffer({
        size: data.byteLength,
        usage: usage | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    if (data instanceof Float32Array) {
        new Float32Array(buffer.getMappedRange()).set(data);
    } else {
        new Uint32Array(buffer.getMappedRange()).set(data);
    }
    buffer.unmap();
    return buffer;
}

function log(msg) {
    document.getElementById("log").innerText += msg + "\n";
    console.log(msg);
}

init();
