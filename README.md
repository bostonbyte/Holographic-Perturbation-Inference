# Holographic Perturbation Inference (HPI)

HPI is an experimental framework for offloading LLM inference to client-side devices by treating the KV-cache as a spectral signal. It utilizes low-rank approximations (SVD) to transfer context state, allowing a WebGPU-based "Resonator" to continue generation locally.

---

## 1. Technical Overview

### Problem: The KV-Cache Bottleneck
In standard Transformer inference, memory requirements for the Key-Value (KV) cache scale with context length, creating a significant memory-bandwidth bottleneck for both server-side hosting and client-side transfer.

### Methodology
1.  **Spectral Compression**: Instead of transferring raw KV tensors, the server performs a Singular Value Decomposition (SVD) on the attention matrices. By retaining only the top $r$ singular values, the semantic state is compressed into a "Hologram."
2.  **Perturbation-Based Inference**: The client approximates the next-token generation as a linear perturbation of the compressed state. This reduces the client-side attention complexity from $O(N^2)$ to $O(1)$ relative to prefill length.

---

## 2. Infrastructure

*   **Anchor (Server)**: A Python FastAPI backend that processes the initial prefill and generates the `.holo` spectral seed (SVD components).
*   **Resonator (Client)**: A WebGPU (WGSL) engine that reconstructs the attention manifold and executes the generation loop locally.
*   **Protocol**: A custom binary format for efficient transmission of $U, \Sigma, V^T$ components in Float16.

---

## 3. Empirical Results

### Quantitative Successes
*   **Compression**: Achieved **90% reduction** in context footprint (Rank 10%) with minimal initial KL-divergence ($<0.005$).
*   **Local Throughput**: Generated **22.4 Tokens/Sec** on consumer-grade hardware (M-series GPUs) using raw WGSL kernels.

### Technical Failures (Numerical Drift Wall)
Despite high architectural efficiency, the system fails to maintain semantic coherence during extended local generation:
1.  **Precision Divergence**: Minor floating-point discrepancies between PyTorch (Server) and WebGPU (Client) kernels compound across 12 transformer layers.
2.  **Logit Explosion**: Numerical drift leads to abnormal logit scales ($>100$). This collapses the Softmax distribution, resulting in repetitive token output (e.g., "the series of the series...").
3.  **Deterministic Collapse**: Even with 100% rank preservation (lossless compression), cross-platform precision differences trigger a "neural brain death" state after few tokens.

---

## 4. Setup & Usage

### Prerequisites
*   Python 3.8+
*   Browser with WebGPU support (Chrome 113+, Edge)

### Installation
1. **Clone repository**.
2. **Install Dependencies**:
   ```bash
   pip install -r server/requirements.txt
   ```
3. **Export Weights**: The WebGPU client requires a local binary of GPT-2 weights.
   ```bash
   python3 server/exporter.py
   ```
   This script downloads the weights from HuggingFace and exports them to `web/gpt2_weights.bin`.

### Execution
1.  **Start Server**: `python3 server/app.py` (Default: `http://localhost:8081`)
2.  **Launch Interface**: Open `web/chat.html`.

---

## 5. Directory Structure
*   `hololib/`: SVD compression and serialization protocol.
*   `server/`: FastAPI prefill engine and weight exporter.
*   `web/`: WebGPU Resonator implementation (WGSL).
*   `experiments/`: Mathematical validation and debugging scripts (SVD, drift, weights).
*   `docs/`: Project background and original planning documents.

## License
MIT

