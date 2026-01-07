export class HoloLoader {
    static async load(url) {
        const buffer = await fetch(url).then(r => r.arrayBuffer());
        return this.parse(buffer);
    }

    static parse(buffer) {
        const view = new DataView(buffer);
        const magic = new TextDecoder().decode(buffer.slice(0, 4));
        if (magic !== 'HOLO') {
            console.error("Received Header:", magic);
            throw new Error(`Invalid Magic Bytes. Expected 'HOLO', got '${magic}'`);
        }

        let offset = 4;
        const version = view.getUint32(offset, true); offset += 4;
        const numLayers = view.getUint32(offset, true); offset += 4;
        const numHeads = view.getUint32(offset, true); offset += 4;
        const headDim = view.getUint32(offset, true); offset += 4;
        const rank = view.getUint32(offset, true); offset += 4;
        const seqLen = view.getUint32(offset, true); offset += 4;

        console.log(`Holo Loaded: v${version} L=${numLayers} H=${numHeads} D=${headDim} R=${rank} S=${seqLen}`);

        // Calculate sizes
        const sizeU = headDim * rank * 2; // bytes
        const sizeS = rank * 2;
        const sizeV = rank * seqLen * 2;

        const layers = [];

        for (let l = 0; l < numLayers; l++) {
            const heads = [];
            for (let h = 0; h < numHeads; h++) {
                // Read chunks
                const chunkU = buffer.slice(offset, offset + sizeU); offset += sizeU;
                const chunkS = buffer.slice(offset, offset + sizeS); offset += sizeS;
                const chunkV = buffer.slice(offset, offset + sizeV); offset += sizeV;

                // Convert to Float32 for WebGPU (unless we use f16 extension)
                heads.push({
                    u: this.decodeF16(new Uint16Array(chunkU)),
                    s: this.decodeF16(new Uint16Array(chunkS)),
                    v: this.decodeF16(new Uint16Array(chunkV))
                });
            }
            layers.push(heads);
        }

        return { layers, numLayers, numHeads, headDim, rank, seqLen };
    }

    // Simple F16 to F32 decoder
    static decodeF16(uint16Arr) {
        const out = new Float32Array(uint16Arr.length);
        for (let i = 0; i < uint16Arr.length; i++) {
            out[i] = this.float16ToFloat32(uint16Arr[i]);
        }
        return out;
    }

    // Standard IEEE 754 half-precision to single-precision
    static float16ToFloat32(h) {
        const s = (h & 0x8000) >> 15;
        const e = (h & 0x7C00) >> 10;
        const f = h & 0x03FF;

        if (e === 0) {
            // Subnormal or zero
            if (f === 0) return s ? -0.0 : 0.0;
            return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
        } else if (e === 0x1F) {
            return f === 0 ? (s ? -Infinity : Infinity) : NaN;
        }
        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }
}
