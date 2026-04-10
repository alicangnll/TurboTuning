#pragma once

#include "llama-kv-cache.h"
#include <memory>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>
#include <unordered_set>

//
// llama_layer_prefetcher (Threads 1-2: Disk -> RAM)
//
// Double-buffered async prefetch pipeline:
//   - 2 worker threads share a job deque (capacity = PREFETCH_LOOKAHEAD)
//   - Prefetches N+1 and N+2 concurrently, eliminating wait stalls during forward pass
//   - On Linux: uses posix_fadvise + readahead() for kernel read-ahead bypass
//   - On Linux: uses MADV_FREE (lazy reclaim) instead of MADV_DONTNEED (immediate zero)
//   - Memory pressure guard: drops lookahead-2 jobs when free RAM < pressure_threshold_mb
//

// How many layers ahead to prefetch (2 = double-buffering)
static constexpr int PREFETCH_LOOKAHEAD = 2;

// How many parallel I/O threads for prefetch
static constexpr int PREFETCH_N_THREADS = 2;

// Helper to unload a specific memory region from physical RAM
void llama_unload_address(void * addr, size_t size);

// Returns free RAM in MB (cross-platform: macOS + Linux)
int64_t llama_get_free_ram_mb();

class llama_layer_prefetcher {
public:
    explicit llama_layer_prefetcher(const struct llama_model & model, int64_t pressure_threshold_mb = 2048);
    ~llama_layer_prefetcher();

    // Enqueue background prefetch for layer `il`.
    // Automatically skips if already in-flight or done.
    void prefetch(int il);

    // Prefetch specific sub-layer parts
    void prefetch_partial(int il, bool attn, bool ffn);

    // Block until layer `il` weights are confirmed ready.
    void wait(int il);

    // Release RAM for layer `il`.
    // Linux: MADV_FREE (lazy) | macOS: MADV_DONTNEED
    void unload(int il);

    // Release specific sub-layer parts
    void unload_partial(int il, bool attn, bool ffn);

    // Global footprint minimization: evacuate all layers from physical RAM.
    void unload_all() const;

private:
    const struct llama_model & model;
    int64_t pressure_threshold_mb;

    std::vector<std::thread> workers;
    std::mutex mtx;
    std::condition_variable cv_enqueue;  // wakes workers when job arrives
    std::condition_variable cv_done;     // wakes waiters when layer is ready

    struct prefetch_job {
        int il = -1;
        bool attn = true;
        bool ffn = true;
    };

    // Ring-buffer job deque, max PREFETCH_LOOKAHEAD pending items
    std::deque<prefetch_job> job_queue;
    std::unordered_set<int> in_flight;   // layers currently being fetched
    std::unordered_set<int> done_set;    // layers fully resident in RAM
    bool should_exit = false;

    void worker_loop();
    void do_prefetch_tensor_hints(struct ggml_tensor * t);
};

//
// llama_kv_compress_worker (Thread 3: RAM -> Compressed RAM)
//
// Asynchronously compresses the KV cache for the previous layer while the GPU
// computes the current layer.
//

class llama_kv_compress_worker {
public:
    llama_kv_compress_worker(const struct llama_context & ctx);
    ~llama_kv_compress_worker();

    // Submit KV tensors (k, v) for background compression.
    // Shape: [n_heads, seq_len, head_dim]
    void compress_async(int il, struct ggml_tensor * k, struct ggml_tensor * v);

    // Block until compression for layer `il` is complete.
    void wait(int il);

    // Block until all pending compression jobs are finished.
    void wait_all();

private:
    // const struct llama_context & ctx;
    std::thread worker;
    std::mutex mtx;
    std::condition_variable cv;
    
    struct job {
        int il;
        struct ggml_tensor * k;
        struct ggml_tensor * v;
    };
    std::queue<job> jobs;
    int completed_layer = -1;
    bool is_working = false;
    bool should_exit = false;

    void worker_loop();
};

//
// llama_tuning_session
//
// Orchestrates the 3-stage pipeline:
//   Thread 1: prefetch layer N+1
//   Thread 2: compute layer N (main thread)
//   Thread 3: compress KV layer N-1
//

struct llama_tuning_session {
    const struct llama_context & ctx;
    std::unique_ptr<llama_layer_prefetcher> prefetcher;
    std::unique_ptr<llama_kv_compress_worker> compressor;

    llama_tuning_session(const struct llama_context & ctx);
    
    // Core pipeline step for layer N
    void step(int il, struct ggml_tensor * k, struct ggml_tensor * v) const;

    // Trigger prefetching for ALL layers (safe to run in parallel with GPU)
    void prefetch_all() const;

    // Trigger KV compression for ALL layers (run AFTER GPU compute)
    void compress_all(const struct llama_memory_i & memory) const;

    // Block until all asynchronous stages are completed for the current batch.
    void wait_all() const;

    void shutdown();
};
