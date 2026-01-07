export class WeightsLoader {
    constructor(device) {
        this.device = device;
        this.index = null;
        this.binaryData = null;
        this.buffers = {}; // Cache map name -> GPUBuffer
        console.log("WeightsLoader V2 (Transposition Fix) Initialized");
    }

    async load(url) {
        console.log(`Downloading weights from ${url}...`);
        const response = await fetch(url);
        const buffer = await response.arrayBuffer();

        console.log("Parsing weights...");
        const view = new DataView(buffer);
        const headerLen = view.getUint32(0, true); // Little endian

        const headerBytes = new Uint8Array(buffer, 4, headerLen);
        const headerStr = new TextDecoder().decode(headerBytes);
        this.index = JSON.parse(headerStr);

        // The payload starts after 4 + headerLen
        const payloadOffset = 4 + headerLen;
        this.binaryData = buffer.slice(payloadOffset);

        console.log(`Weights Loaded. Index has ${Object.keys(this.index).length} tensors.`);

        // Debug: Verify a few decoded weights
        const testName = 'transformer.wte.weight';
        const testMeta = this.index[testName];
        if (testMeta) {
            const slice = this.binaryData.slice(testMeta.offset, testMeta.offset + 20); // First 10 f16 values
            const f16 = new Uint16Array(slice);
            const f32 = this.decodeF16(f16);
            console.log(`DEBUG: First 10 wte.weight values (f16 raw):`, Array.from(f16));
            console.log(`DEBUG: First 10 wte.weight values (decoded f32):`, Array.from(f32));
        }
    }

    transpose(data, rows, cols) {
        const out = new Float32Array(data.length);
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                // Source: [r, c] (Row Major: r * cols + c)
                // Dest:   [c, r] (Row Major: c * rows + r)
                out[c * rows + r] = data[r * cols + c];
            }
        }
        return out;
    }

    getTensor(name) {
        if (this.buffers[name]) return this.buffers[name];

        const metadata = this.index[name];
        if (!metadata) throw new Error(`Tensor ${name} not found`);

        // Extract Float16 bytes
        const start = metadata.offset;
        const end = start + metadata.size;
        const slice = this.binaryData.slice(start, end);

        // Convert to Float32
        const f16 = new Uint16Array(slice);
        let f32 = this.decodeF16(f16);

        // CHECK TRANSPOSE for Conv1D layers (In, Out) -> (Out, In)
        // GPT-2 config: n_embd = 768.
        // matching names: c_attn, c_proj, c_fc

        let shouldTranspose = false;
        let inDim = 0;
        let outDim = 0;

        if (name.includes('attn.c_attn.weight')) {
            inDim = 768; outDim = 2304; shouldTranspose = true;
        } else if (name.includes('attn.c_proj.weight')) {
            inDim = 768; outDim = 768; shouldTranspose = true;
        } else if (name.includes('mlp.c_fc.weight')) {
            inDim = 768; outDim = 3072; shouldTranspose = true;
        } else if (name.includes('mlp.c_proj.weight')) {
            inDim = 3072; outDim = 768; shouldTranspose = true;
        }

        if (shouldTranspose) {
            console.log(`Transposing ${name} (${inDim}x${outDim} -> ${outDim}x${inDim})`);
            f32 = this.transpose(f32, inDim, outDim);
        }

        // Upload to GPU
        const gpuBuffer = this.device.createBuffer({
            label: name,
            size: f32.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(gpuBuffer.getMappedRange()).set(f32);
        gpuBuffer.unmap();

        this.buffers[name] = gpuBuffer;
        return gpuBuffer;
    }

    decodeF16(uint16Arr) {
        const out = new Float32Array(uint16Arr.length);
        for (let i = 0; i < uint16Arr.length; i++) {
            out[i] = this.float16ToFloat32(uint16Arr[i]);
        }
        return out;
    }

    float16ToFloat32(h) {
        const s = (h & 0x8000) >> 15;
        const e = (h & 0x7C00) >> 10;
        const f = h & 0x03FF;

        if (e === 0) {
            // Subnormal or zero
            if (f === 0) return s ? -0.0 : 0.0;
            // Subnormal: (-1)^s * 2^(-14) * (f/1024)
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        }
        if (e === 0x1F) {
            return f === 0 ? (s ? -Infinity : Infinity) : NaN;
        }
        // Normal number: (-1)^s * 2^(e-15) * (1 + f/1024)
        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }
}
