# TurboQuant+ // LLMTuning Roadmap 🗺️

This document outlines the development trajectory of the 2026 Engine, from its mathematical origins to its current state as a hardware-aware orchestration layer.

---

## 🔬 Phase I: The Mathematics (TurboQuant Research)
**Focus**: Proving the Distortion-Rate bounds of PolarQuant.
*Status: Completed*

- [x] **PolarQuant Prototype**: Developed Python scripts to validate Walsh-Hadamard rotations and Beta-distribution quantization [[2]](https://arxiv.org/abs/2504.19874).
- [x] **Distortion Benchmarking**: Verified that 2-bit PolarQuant matches the performance of 4-bit standard scalar quantization.
- [x] **Sparse V Theory**: Designed the attention-gating mechanism to skip Value-tensor dequantization for low-weight tokens.

---

## 🛠️ Phase II: The Engine (LLMa Integration)
**Focus**: Porting research to high-performance C++ and `llama.cpp`.
*Status: Completed*

- [x] **C++ Implementation**: Ported Walsh-Hadamard transforms to optimized C loops with OpenMP acceleration.
- [x] **Metal Kernel Development**: Integrated 2-bit/3-bit/4-bit KV dequantization directly into Apple Metal shaders.
- [x] **Repack Suppression**: Identified and disabled redundant CPU memory buffers to achieve the **1.1GB cold-start** milestone.

---

## 🧠 Phase III: The Orchestrator (LLMTuning)
**Focus**: Solving the physical RAM barrier via virtualization.
*Status: Active / Stable*

- [x] **Active Sharding**: Implemented `madvise(DONTNEED)` logic to release transformer layers from RAM immediately after use.
- [x] **Predictive Paging**: Added background prefetching threads to overlap SSD I/O with GPU computation.
- [x] **Sub-Layer Pipelining**: Split each layer into Attn/FFN halves for finer-grained prefetch/evict via `prefetch_partial()` and `unload_partial()`.
- [x] **Native Budgeting**: Implemented hardware discovery for macOS (Smart-Tighten 2.8) to auto-calculate layer budgets.
- [x] **TQR Hijacking**: Developed the TurboQuant Repack (`.tqr`) format for instant weight hot-swapping on startup, with auto-cleanup of stale TQR files.
- [x] **Auto-Activation**: LLMTuning now activates automatically when any turbo cache type is detected — no separate `--turbo-async` flag needed.
- [x] **Validation Pass**: Demo scripts run a quick 50-token LLMTuning validation before the main interactive session to verify pipeline integrity.
- [x] **Model Portfolio Expansion**: Added GPT 20B (OpenAI OSS), Gemma 4 31B (Google), and Qwen 2.5 Coder 7B to demo scripts.

---

## 🚀 Phase IV: Future Horizons
**Focus**: Performance scaling and cross-platform dominance.
*Status: Planned*

### 1. Multi-GPU Sharding
*   **Goal**: Orchestrate weight swapping across multiple asymmetrical GPUs (e.g., an internal Mac GPU + an external eGPU).
*   **Target**: 500B+ models running on mixed-hardware setups.

### 2. Kernel-Level Paging
*   **Goal**: Develop a specialized Linux kernel module (and macOS KEXT/Driver equivalent) for high-frequency weight swapping.
*   **Target**: Reduce the performance overhead of `madvise` context switching.

### 3. Real-Time KV Pruning
*   **Goal**: Dynamically prune attention heads that contribute zero to the output, further reducing the KV cache footprint during long-context generation.

### 4. Linux Native Budget Discovery
*   **Goal**: Implement `/proc/meminfo` parsing for `llama_get_device_memory_budget()` (currently falls back to 16384 MB on Linux).

### 5. Unified Python Orchestration
*   **Goal**: Extend `cli_config_export.py` usage to macOS/Linux demo scripts (currently Windows-only).

---

## ✅ Success Criteria
1.  **Transparency**: The engine must accurately report physical vs. virtual RAM usage to the user.
2.  **Safety**: "Zero-Spike" initialization must prevent system-wide freezes on low-memory devices.
3.  **Accuracy**: PPL metrics must stay within 0.1% of the floating-point baseline despite extreme quantization.

---

> [!TIP]
> **Contribute**: We welcome research papers and PRs focused on Walsh-Hadamard optimizations and memory paging strategies.

---

## 📚 References

1. Turney, T. (2026). *TurboQuant+: Extreme-Efficiency Inference Engine for Large Language Models*. GitHub repository. [https://github.com/TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
2. *PolarQuant Algorithm*. ICLR 2026. (arXiv:2504.19874)
