#!/usr/bin/env python3
import subprocess
import re
import os
import sys
import platform
import argparse
from typing import Dict, Optional, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def get_time_cmd() -> List[str]:
    """Return the platform-specific wrapper command to measure memory usage."""
    if platform.system() == "Darwin":
        return ["/usr/bin/time", "-l"]
    else:
        return ["/usr/bin/time", "-v"]

def can_run_time_cmd() -> bool:
    """Check if the time command exists on the OS."""
    return os.path.exists(get_time_cmd()[0])

def run_benchmark(name: str, cmd: List[str], env: dict) -> Dict[str, float]:
    print(f"\n--- Running {name} Benchmark ---")
    print(f"> Command: '{' '.join(cmd)}'")
    
    # Wrap with time command for memory tracking if available
    time_cmd = get_time_cmd()
    full_cmd = time_cmd + cmd if can_run_time_cmd() else cmd
    
    # Let's show stdout dynamically if desired, or capture it heavily
    # We will capture it to parse the timing and memory numbers.
    process = subprocess.Popen(
        full_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    # communicate() waits for completion and gets stdout/stderr
    # Send /exit to stdin to ensure llama-cli exits if it defaults to interactive mode
    stdout, stderr = process.communicate(input="/exit\n")
    output = stdout + stderr
    
    if process.returncode != 0:
        print(f"Warning: {name} command exited with non-zero exit status {process.returncode}")
        # Not exiting early, we might still have useful stats to parse
        
    results = {}
    
    # Parse evaluation metrics
    # e.g., llama_print_timings: prompt eval time =    1234.56 ms /    10 tokens (  123.45 ms per token,     8.10 tokens per second)
    prompt_eval_re = re.search(r"prompt eval time.*?=\s*[\d.]+\s*ms\s*/.*?,\s*([\d.]+)\s*tokens per second", output)
    if prompt_eval_re:
        results["prompt_eval_tokens_per_sec"] = float(prompt_eval_re.group(1))
    else:
        # Fallback to the interactive status indicator e.g. [ Prompt: 178.9 t/s | Generation: 19.5 t/s ]
        prompt_alt_re = re.search(r"\[ Prompt:\s+([\d.]+)\s+t/s", output)
        if prompt_alt_re:
            results["prompt_eval_tokens_per_sec"] = float(prompt_alt_re.group(1))

    # e.g., llama_print_timings:        eval time =    1234.56 ms /    10 runs   (  123.45 ms per token,     8.10 tokens per second)
    eval_re = re.search(r"eval time.*?=\s*[\d.]+\s*ms\s*/.*?,\s*([\d.]+)\s*tokens per second", output)
    if eval_re:
        results["eval_tokens_per_sec"] = float(eval_re.group(1))
    else:
        # Fallback to the interactive status indicator e.g. [ Prompt: 178.9 t/s | Generation: 19.5 t/s ]
        eval_alt_re = re.search(r"\|\s*Generation:\s+([\d.]+)\s+t/s", output)
        if eval_alt_re:
            results["eval_tokens_per_sec"] = float(eval_alt_re.group(1))
        
    # Parse memory sizes from llama log output to give fallback metrics
    calc_re = re.search(r"compute buffer total size =\s*([\d.]+)\s*MiB", output)
    if calc_re:
        results["llama_compute_buffer_mb"] = float(calc_re.group(1))
        
    # Parse memory metrics from system `time` wrapper
    if can_run_time_cmd():
        if platform.system() == "Darwin":
            # macOS format: "123456  maximum resident set size"
            mem_re = re.search(r"^\s*(\d+)\s+maximum resident set size", output, re.MULTILINE)
            if mem_re:
                # macOS reports in bytes, converting to MB
                results["max_rss_mb"] = float(mem_re.group(1)) / (1024 * 1024)
        else:
            # Linux format: "Maximum resident set size (kbytes): 123456"
            mem_re = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", output, re.IGNORECASE)
            if mem_re:
                results["max_rss_mb"] = float(mem_re.group(1)) / 1024

    return results

def print_results_table(all_results: Dict[str, Dict[str, float]], base_name: str, tt_name: str):
    print("\n" + "="*60)
    print(" " * 20 + "BENCHMARK RESULTS")
    print("="*60)
    print(f"{'Metric':<25} | {base_name:<15} | {tt_name:<15}")
    print("-" * 60)
    
    metrics = [
        ("prompt_eval_tokens_per_sec", "Prefill (t/s)"),
        ("eval_tokens_per_sec", "Generation (t/s)"),
        ("llama_compute_buffer_mb", "Compute Buffer (MB)"),
        ("max_rss_mb", "Peak Sys Memory (MB)")
    ]
    
    for key, display_name in metrics:
        base_val = all_results[base_name].get(key)
        tt_val = all_results[tt_name].get(key)
        
        base_str = f"{base_val:.2f}" if base_val is not None else "N/A"
        tt_str = f"{tt_val:.2f}" if tt_val is not None else "N/A"
        
        print(f"{display_name:<25} | {base_str:<15} | {tt_str:<15}")
    print("="*60 + "\n")

def plot_results(all_results: Dict[str, Dict[str, float]], base_name: str, tt_name: str):
    if not MATPLOTLIB_AVAILABLE:
        print("\nNote: 'matplotlib' is not installed. To generate a scientific graph, please run:")
        print("  pip install matplotlib numpy")
        print("and run this benchmark again.\n")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Speed Chart ---
    labels_speed = ['Prefill (t/s)', 'Generation (t/s)']
    base_speed = [all_results[base_name].get('prompt_eval_tokens_per_sec', 0),
                  all_results[base_name].get('eval_tokens_per_sec', 0)]
    tt_speed = [all_results[tt_name].get('prompt_eval_tokens_per_sec', 0),
                all_results[tt_name].get('eval_tokens_per_sec', 0)]
                
    x = np.arange(len(labels_speed))
    width = 0.35
    
    ax1.bar(x - width/2, base_speed, width, label=base_name, color='#1f77b4', edgecolor='black')
    ax1.bar(x + width/2, tt_speed, width, label=tt_name, color='#ff7f0e', edgecolor='black')
    
    ax1.set_ylabel('Tokens per Second (Higher is Better)')
    ax1.set_title('Inference Speed Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_speed)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Memory Chart ---
    labels_mem = ['Peak Sys Mem (MB)']
    # Fallback to compute buffer if max_rss_mb is missing
    base_mem = [all_results[base_name].get('max_rss_mb') or all_results[base_name].get('llama_compute_buffer_mb', 0)]
    tt_mem = [all_results[tt_name].get('max_rss_mb') or all_results[tt_name].get('llama_compute_buffer_mb', 0)]
    
    x_mem = np.arange(len(labels_mem))
    ax2.bar(x_mem - width/2, base_mem, width, label=base_name, color='#1f77b4', edgecolor='black')
    ax2.bar(x_mem + width/2, tt_mem, width, label=tt_name, color='#ff7f0e', edgecolor='black')
    
    ax2.set_ylabel('Memory in MB (Lower is Better)')
    ax2.set_title('Memory Footprint Comparison')
    ax2.set_xticks(x_mem)
    ax2.set_xticklabels(labels_mem)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.suptitle('TurboTuning vs Baseline Performance', y=1.05, fontsize=14, fontweight='bold')
    
    output_filename = "benchmark_results.png"
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"=> Scientific benchmark graph successfully saved to: {output_filename}\n")

def main():
    parser = argparse.ArgumentParser(description="TurboTuning vs Baseline Local Benchmark")
    
    # Make sure default paths are relative to where the benchmark script typically gets executed from.
    # Usually `benchmark/` if cd'd into it, or project root.
    
    # Auto-detect root directory 
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    default_cli = os.path.join(project_root, "llama-cpp-turboquant", "build", "bin", "llama-cli")
    default_model = os.path.join(project_root, "models", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

    parser.add_argument("--model", type=str, default=default_model, help=f"Path to the GGUF model (default: {default_model})")
    parser.add_argument("--prompt", type=str, default="Write a 300 word essay about the future of artificial intelligence in space exploration. Be creative and very detailed.", help="Prompt for benchmarking")
    parser.add_argument("--n-predict", type=int, default=256, help="Number of tokens to generate")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="Number of threads")
    parser.add_argument("--ngl", type=int, default=99, help="Number of GPU layers (for offloading)")
    parser.add_argument("--llama-cli", type=str, default=default_cli, help="Path to llama-cli executable")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.llama_cli):
        print(f"Error: llama-cli not found at '{args.llama_cli}'.")
        print("Please compile TurboTuning's cpp engine first.")
        sys.exit(1)
        
    if not os.path.exists(args.model):
        print(f"Error: Model not found at '{args.model}'.")
        print("Please run one of the `run_turboquant_demo_*.sh` scripts to auto-download a model, or provide a path via --model.")
        sys.exit(1)
        
    base_cmd = [
        args.llama_cli,
        "-m", args.model,
        "-p", args.prompt,
        "-n", str(args.n_predict),
        "-c", str(args.n_ctx),
        "-t", str(args.threads),
        "-ngl", str(args.ngl),
        "--no-display-prompt"
    ]
    
    env = os.environ.copy()
    
    # Config keys
    baseline_key = "Baseline"
    tt_key = "TurboTuning"

    run_configs = {
        baseline_key: base_cmd,
        tt_key: base_cmd + [
            "--cache-type-k", "turbo4",
            "--cache-type-v", "turbo3",
            "--turbo-async"
        ]
    }
    
    all_results = {}
    for name, cmd in run_configs.items():
        all_results[name] = run_benchmark(name, cmd, env)
        
    print_results_table(all_results, baseline_key, tt_key)
    plot_results(all_results, baseline_key, tt_key)

if __name__ == "__main__":
    main()
