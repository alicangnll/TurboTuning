# TurboTuning — Architectural Map

This document describes the internal architecture of the two TurboTuning modules — **TurboQuant+** [[1]](https://github.com/TheTom/turboquant_plus) (KV cache compression) and **LLMTuning** (weight memory virtualization) — their data flows, concurrency model, and integration points.

---

## 1. System Overview

```
User Prompt
     │
     ▼
┌─────────────────────────────────────────┐
│       llama-cli / llama-server          │
│   (llama-cpp-turboquant fork)           │
└──────┬──────────────────────┬───────────┘
       │                      │
       ▼                      ▼
┌─────────────────┐  ┌────────────────────────┐
│   LLMTuning     │  │     TurboQuant+        │
│                 │  │                        │
│ 2× prefetch     │  │ WHT rotation (O(d·logd)│
│ threads         │  │ + 2/3/4-bit quant on   │
│ MADV_WILLNEED   │  │ K and V tensors        │
│ MADV_FREE /     │  │                        │
│ MADV_DONTNEED   │  │ KV cache compression   │
│ Weight paging   │  │ 2.5–6.4× vs f16        │
└────────┬────────┘  └──────────┬─────────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         Metal / CUDA / CPU
         Flash Attention kernel
```

---

## 2. TurboQuant+: Data Flow

### 2.1 Quantization — Write Path (storing K/V into cache)

```
Attention K tensor  [f16, natural space]
         │
         ├── turbo2 / turbo3 ─────────────────────────────────┐
         │   turbo_cpu_fwht() on CPU                          │
         │   or kernel_set_rows_turbo2/3 on Metal             │
         │   → K stored in WHT-rotated domain                 │
         │                                                    │
         └── turbo4 ──────────────────────────────────────────┤
             WHT skipped                                      │
             kernel_set_rows_turbo4 quantizes directly        │
             → K stored in natural domain                     │
                                                              ▼
                                              KV Cache  [compressed, RAM]
```

### 2.2 Dequantization — Read Path (Flash Attention)

```
KV Cache  [compressed]
         │
         ├── turbo2 / turbo3 K ──────────────────────────────┐
         │   Centroid lookup, stays in WHT-rotated domain    │
         │                                                   │
         └── turbo4 K ─────────────────────────────────────┐ │
             Centroid lookup, stays in natural domain       │ │
                                                            │ │
             Flash Attention: softmax(Q @ K^T / √d) @ V ◄──┘ │
             Metal non-vec kernel (always used for turbo) ◄───┘
```

### 2.3 Q Domain Matching (`llama-graph.cpp`)

```
Q tensor  [natural space, from model]
         │
         ├── K is turbo2 or turbo3 ──► WHT applied to Q ──► Q_rotated @ K_rotated^T  ✓
         │
         └── K is turbo4 ────────────► Q unchanged ────────► Q_natural @ K_natural^T  ✓
```

`llama-graph.cpp` inspects `k->type` at each layer and conditionally applies `turbo_cpu_fwht()` or the Metal WHT kernel to Q. Mismatched domains produce garbage attention scores — the check is mandatory.

### 2.4 Inverse WHT on Attention Output

```
FA output  [in V's domain]
         │
         ├── V is turbo2 or turbo3 ──► inverse WHT applied ──► output in natural space  ✓
         │
         └── V is turbo4 ────────────► output unchanged ───────► output in natural space  ✓
```

---

## 3. LLMTuning: Concurrency Model

### 3.1 Three-Stage Async Pipeline

`llama_tuning_session` owns three concurrent actors:

```
┌──────────────────────────────────────────────────────────────────┐
│                    llama_tuning_session                          │
│                                                                  │
│  Threads 1–2: Prefetcher        Thread 0: GPU        Thread 3:  │
│  (llama_layer_prefetcher)       (main thread)        KV Worker  │
│  ─────────────────────          ─────────────        ─────────  │
│  job_queue depth = 2            Executes layer N     f16 → turbo │
│  2 worker threads                                   in-place    │
│                                                     async       │
│  MADV_WILLNEED N+1, N+2                                         │
│  MADV_FREE / DONTNEED N-1                                        │
└──────────────────────────────────────────────────────────────────┘
```

Timeline for a single decode step:

```
Time ──────────────────────────────────────────────────────────►

GPU:      [  Compute layer N  ]  [  Compute N+1  ]
I/O:   [Prefetch N+1][Prefetch N+2]  [Prefetch N+2][Prefetch N+3]
Evict: [Free N-1]                    [Free N]
KV:    [Compress KV(N-1)]            [Compress KV(N)]
```

### 3.2 Double-Buffered Prefetcher

**Key data structures:**

```cpp
// Ring-buffer job queue (capacity = PREFETCH_LOOKAHEAD = 2)
std::deque<prefetch_job>     job_queue;
std::unordered_set<int>      in_flight;   // layers being fetched
std::unordered_set<int>      done_set;    // layers in page cache

// Two worker threads — both pull from job_queue
std::vector<std::thread>     workers;     // size = PREFETCH_N_THREADS = 2

// Condition variables: one for job arrival, one for completion
std::condition_variable      cv_enqueue;
std::condition_variable      cv_done;
```

**Memory pressure guard:** Before each `prefetch_partial()` call, `llama_get_free_ram_mb()` is queried. If free RAM < `pressure_threshold_mb`, the farther lookahead (N+2) is dropped. The threshold is set adaptively at session init: 1 GB if total free RAM < 4 GB, 2 GB otherwise.

### 3.3 Platform-Specific Eviction

| Platform | Flag | Behavior |
|---|---|---|
| macOS | `MADV_DONTNEED` | Immediate physical page release; virtual mapping preserved |
| Linux | `MADV_FREE` | Lazy release — kernel reclaims only under pressure; no disk re-read if still warm |

The flag is selected at compile time via `llama_madv_unload_flag()`. On Metal (Apple Silicon), weight tensors backed by `newBufferWithBytesNoCopy` share physical pages with the mmap'd file. The unload function skips any tensor where `!ggml_backend_buffer_is_host(t->buffer)` to prevent nil-buffer crashes in Metal.

### 3.4 KV Compression Worker — In-Place Quantization

For KV caches initially allocated as f16, the background worker performs in-place compression:

```
f16 tensor (host buffer)
     │
     ├─ 1. Decode f16 → f32  (temporary staging vector)
     │
     ├─ 2a. K → quantize_turbo4_0()   [K: less precision-sensitive]
     │   2b. V → quantize_turbo3_0()  [V: moderate compression]
     │
     └─ 3. Update t->type, t->nb[0..3]
          (compressed bytes always fit: turbo < f16)
```

Metal/CUDA buffers are detected via `ggml_backend_buffer_is_host()` and skipped. Only host-accessible (CPU-mapped) tensors are compressed in-place.

### 3.5 TQR Boot Optimization

```
Session init
     │
     ├── ggml_cpu_repack_is_hijacked()?
     │     YES → Zero-Allocation mode (pre-mapped pages, no copy)
     │
     └── NO → <model_desc>.tqr exists?
               │
               ├── YES → llama_model_repack_load()
               │          mmap page-aligned weights → cold start ~1.1 GB
               │
               └── NO → llama_model_repack_save()
                          write page-aligned .tqr
                          auto-clean stale .tqr files for other models
```

TQR stores tensors at page-aligned offsets (4 KB boundaries). On subsequent runs, `mmap` maps them directly with `MAP_SHARED`, eliminating the temporary repack buffer that standard GGUF loading allocates (~500 MB–2 GB for large models).

### 3.6 Memory Timeline

```
T=0  Startup:
       model mmap'd → MADV_DONTNEED all layers    →  RSS ≈ 1.1 GB  (embed + output only)

T=1  First token, Layer 0:
       Prefetch 1, 2 (async, 2 threads)           →  RSS ≈ 1.4 GB
       GPU computes 0
       MADV_FREE / DONTNEED layer 0               →  RSS ≈ 1.1 GB

T=2  Layer 1:
       Already in page cache (prefetched at T=1)
       GPU computes 1
       MADV_FREE layer 1, prefetch 3              →  RSS ≈ 1.1–1.4 GB

... repeats for all N layers ...

Steady state:  1–2 layers physically resident  ≈  (1–2) × layer_size
               For 70B Q4_K_M: layer_size ≈ 300 MB  →  steady RSS ≈ 0.6–1.2 GB weights
```

---

## 4. Memory Budget Formula

Used by `cli_config_export.py` to derive `ngl` (GPU layers) and context length:

```
budget_mb  =  hw.memsize × 0.75   (macOS Metal reservation)

kv_mb      =  ctx_len × n_layers × n_heads × head_dim × 2   (f16 baseline)
           ×  (100 / compression_ratio)                       (turbo scaling)

graph_mb   =  clamp(model_size_mb / 2,  100,  5000)

weight_mb  =  (budget_mb − kv_mb − graph_mb) × MEM_BUDGET_PCT

ngl        =  floor(weight_mb / bytes_per_layer)
```

KV compression ratios (×100 integer arithmetic):

| Type | Ratio | Effective size vs f16 |
|---|---|---|
| `turbo4` | 376 | 26.6% |
| `turbo3` | 490 | 20.4% |
| `turbo2` | 640 | 15.6% |

Memory tiers:

| Tier | `MEM_BUDGET_PCT` | `batch` | `ubatch` | V cache |
|---|---|---|---|---|
| 1 — Performance | 90% | 512 | 256 | turbo4 |
| 2 — Balanced | 40% | 256 | 128 | turbo4 |
| 3 — Ultra-Eco | 10% | 32 | 32 | turbo2 |

---

## 5. Source File Map

### TurboQuant+ Core

| File | Purpose |
|---|---|
| `ggml/src/ggml-turbo-quant.c` | CPU quantize/dequantize for turbo2/3/4 |
| `ggml/src/ggml-metal/ggml-metal.metal` | 537 Metal kernel specializations |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | Metal dispatch — vec vs non-vec FA path |
| `src/llama-graph.cpp` | Q WHT rotation; output inverse WHT |
| `src/llama-kv-cache.cpp` | KV cache storage and retrieval |

### LLMTuning

| File | Purpose |
|---|---|
| `src/llama-llmtuning.cpp` | Prefetcher, KV compressor, tuning session |
| `src/llama-llmtuning.h` | `llama_layer_prefetcher`, `llama_kv_compress_worker`, `llama_tuning_session` |
| `src/llama-repack.cpp` | TQR save/load — page-aligned `.tqr` files |
| `src/llama-context.cpp` | Context init — creates `tuning_session` when turbo detected |

### Orchestration

| File | Purpose |
|---|---|
| `common/arg.cpp` | Argument parsing — sets `turbo_async=true` for turbo cache types |
| `turboquant/cli_config_export.py` | Memory budget → `tq_cli_config.json` |
| `turboquant/layer_profiler.py` | Per-layer adaptive quantization profiler |

---

## 6. Standalone Usage

### TurboQuant+ without LLMTuning

```bash
./build/bin/llama-cli \
    -m model.gguf \
    -ngl 99 \
    -c 4096 \
    --cache-type-k turbo4 \
    --cache-type-v turbo4
```

Use when the model fits in VRAM/RAM. Only KV memory is reduced.

### Asymmetric K/V types

```bash
--cache-type-k turbo4 --cache-type-v turbo2
```

K uses turbo4 (higher fidelity dot product). V uses turbo2 (maximum output memory savings). Saves an additional ~15% compared to symmetric turbo4/turbo4.

---

## 7. References

1. Turney, T. (2026). *TurboQuant+: Extreme-Efficiency Inference Engine for Large Language Models*. GitHub repository. [https://github.com/TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
2. *PolarQuant Algorithm*. ICLR 2026. (arXiv:2504.19874)
