# TurboTuning & TurboQuant+ Benchmark

This directory contains benchmarking tools to evaluate the performance differences (speed and memory) between the baseline inference engine and TurboTuning (which includes **TurboQuant+** KV compression and **LLMTuning** weight virtualization).

## Quick Start

You can run the benchmark script directly. It will automatically detect the paths if run from anywhere within the repository. It uses your native Python 3 library without any external dependencies.

```bash
cd benchmark
python3 benchmark.py
```

## How It Works

The `benchmark.py` script runs `llama-cli` two times:
1. **Baseline**: Standard generation with 16-bit KV cache and no LLMTuning optimizations.
2. **TurboTuning**: With `--cache-type-k turbo4`, `--cache-type-v turbo3`, and `--turbo-async` enabled. This activates the TurboQuant+ quantized KV cache and the LLMTuning memory virtualization pipeline.

It tracks:
- **Prefill (t/s)**: Time to process the initial prompt. 
- **Generation (t/s)**: Token generation speed.
- **Compute Buffer (MB)**: Space pre-allocated by the engine for compute nodes.
- **Peak Sys Memory (MB)**: Total Max RSS (Resident Set Size) memory footprint at the OS level (monitored via `time`).

## Arguments

```bash
usage: benchmark.py [-h] [--model MODEL] [--prompt PROMPT] [--n-predict N_PREDICT] 
                     [--n-ctx N_CTX] [--threads THREADS] [--ngl NGL] [--llama-cli LLAMA_CLI]
```

- `--model`: Path to standard GGUF model files. Default falls back to `models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` if available.
- `--prompt`: Custom benchmark prompt. Default is a ~300-word generation prompt.
- `--n-predict`: Amount of generated tokens.
- `--n-ctx`: Context window size. Increasing this highlights TurboTuning memory advantage dynamically.
- `--ngl`: Number of GPU layers to offload. Default `99`.
- `--threads`: Compute thread count.

## Example Output

```
============================================================
                    BENCHMARK RESULTS
============================================================
Metric                    | Baseline        | TurboTuning    
------------------------------------------------------------
Prefill (t/s)             | 85.12           | 84.77          
Generation (t/s)          | 24.31           | 25.10          
Compute Buffer (MB)       | 512.00          | 512.00         
Peak Sys Memory (MB)      | 6140.23         | 1342.11        
============================================================
```
