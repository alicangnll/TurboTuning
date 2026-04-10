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
    
    time_cmd = get_time_cmd()
    full_cmd = time_cmd + cmd if can_run_time_cmd() else cmd
    
    process = subprocess.Popen(
        full_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    # Send /exit to stdin to ensure llama-cli exits if it defaults to interactive mode
    stdout, stderr = process.communicate(input="/exit\n")
    raw_output = stdout + stderr
    
    # Strip ANSI escape codes to ensure clean regex matching
    output = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', raw_output)
    
    if process.returncode != 0:
        print(f"Warning: {name} command exited with non-zero exit status {process.returncode}")
        
    results = {}
    
    prompt_eval_re = re.search(r"prompt eval time.*?=\s*[\d.]+\s*ms\s*/.*?,\s*([\d.]+)\s*tokens per second", output)
    if prompt_eval_re:
        results["prompt_eval_tokens_per_sec"] = float(prompt_eval_re.group(1))
    else:
        prompt_alt_re = re.search(r"\[ Prompt:\s+([\d.]+)\s+t/s", output)
        if prompt_alt_re:
            results["prompt_eval_tokens_per_sec"] = float(prompt_alt_re.group(1))

    eval_re = re.search(r"eval time.*?=\s*[\d.]+\s*ms\s*/.*?,\s*([\d.]+)\s*tokens per second", output)
    if eval_re:
        results["eval_tokens_per_sec"] = float(eval_re.group(1))
    else:
        eval_alt_re = re.search(r"\|\s*Generation:\s+([\d.]+)\s+t/s", output)
        if eval_alt_re:
            results["eval_tokens_per_sec"] = float(eval_alt_re.group(1))
        
    calc_re = re.search(r"compute buffer total size =\s*([\d.]+)\s*MiB", output)
    if calc_re:
        results["llama_compute_buffer_mb"] = float(calc_re.group(1))
        
    if can_run_time_cmd():
        if platform.system() == "Darwin":
            mem_re = re.search(r"^\s*(\d+)\s+maximum resident set size", output, re.MULTILINE)
            if mem_re:
                results["max_rss_mb"] = float(mem_re.group(1)) / (1024 * 1024)
        else:
            mem_re = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", output, re.IGNORECASE)
            if mem_re:
                results["max_rss_mb"] = float(mem_re.group(1)) / 1024

    return results

def print_results_table(all_results: Dict[str, Dict[str, float]], config_names: List[str]):
    print("\n" + "="*95)
    print(" " * 35 + "BENCHMARK RESULTS")
    print("="*95)
    
    header = f"{'Metric':<20} | " + " | ".join([f"{name:<15}" for name in config_names])
    print(header)
    print("-" * len(header))
    
    metrics = [
        ("prompt_eval_tokens_per_sec", "Prefill (t/s)"),
        ("eval_tokens_per_sec", "Generation (t/s)"),
        ("llama_compute_buffer_mb", "Compute Buf(MB)"),
        ("max_rss_mb", "Peak Mem (MB)")
    ]
    
    for key, display_name in metrics:
        row = f"{display_name:<20} | "
        for name in config_names:
            val = all_results[name].get(key)
            val_str = f"{val:.2f}" if val is not None else "N/A"
            row += f"{val_str:<15} | "
        print(row[:-3])
    print("="*95 + "\n")

def plot_results(all_results: Dict[str, Dict[str, float]], config_names: List[str], model_path: str):
    if not MATPLOTLIB_AVAILABLE:
        print("\nNote: 'matplotlib' is not installed. To generate a scientific graph, please run:")
        print("  pip install matplotlib numpy")
        print("and run this benchmark again.\n")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Custom colors
    colors = ['#7f7f7f', '#1f77b4', '#2ca02c', '#ff7f0e']
    
    # --- Speed Chart ---
    labels_speed = ['Prefill (t/s)', 'Generation (t/s)']
    x_speed = np.arange(len(labels_speed))
    width = 0.2
    
    for i, name in enumerate(config_names):
        speed_vals = [all_results[name].get('prompt_eval_tokens_per_sec', 0),
                      all_results[name].get('eval_tokens_per_sec', 0)]
        offset = width * i - (width * len(config_names)/2) + width/2
        ax1.bar(x_speed + offset, speed_vals, width, label=name, color=colors[i % len(colors)], edgecolor='black')
        
    ax1.set_ylabel('Tokens per Second (Higher is Better)')
    ax1.set_title('Inference Speed Comparison')
    ax1.set_xticks(x_speed)
    ax1.set_xticklabels(labels_speed)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Memory Chart ---
    labels_mem = ['Peak System RAM (MB)']
    x_mem = np.arange(len(labels_mem))
    
    for i, name in enumerate(config_names):
        mem_val = all_results[name].get('max_rss_mb') or all_results[name].get('llama_compute_buffer_mb', 0)
        offset = width * i - (width * len(config_names)/2) + width/2
        ax2.bar(x_mem + offset, [mem_val], width, label=name, color=colors[i % len(colors)], edgecolor='black')
        
    ax2.set_ylabel('Memory in MB (Lower is Better)')
    ax2.set_title('Memory Footprint Comparison')
    ax2.set_xticks(x_mem)
    ax2.set_xticklabels(labels_mem)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    basename = os.path.basename(model_path).lower()
    match = re.search(r'([\d.]+b)', basename)
    model_size = match.group(1).upper() if match else "Unknown"
    title_suffix = f" ({model_size} Model)" if model_size != "Unknown" else ""
    
    plt.suptitle(f'TurboTuning Architecture Benchmarks{title_suffix}', y=1.05, fontsize=15, fontweight='bold')
    
    filename_size = f"_{model_size.lower()}" if model_size != "Unknown" else ""
    output_filename = f"benchmark_results{filename_size}.png"
    
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"=> Scientific benchmark graph successfully saved to: {output_filename}\n")

def main():
    parser = argparse.ArgumentParser(description="TurboTuning Modular Benchmark")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    default_cli = os.path.join(project_root, "llama-cpp-turboquant", "build", "bin", "llama-cli")
    default_model = os.path.join(project_root, "models", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

    parser.add_argument("--model", type=str, default=default_model, help="Path to the GGUF model")
    parser.add_argument("--prompt", type=str, default="Write a 300 word essay about the future of artificial intelligence in space exploration. Be creative and very detailed.", help="Prompt for benchmarking")
    parser.add_argument("--n-predict", type=int, default=256, help="Number of tokens to generate")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 4, help="Number of threads")
    parser.add_argument("--ngl", type=int, default=99, help="Number of GPU layers (for offloading)")
    parser.add_argument("--llama-cli", type=str, default=default_cli, help="Path to llama-cli executable")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.llama_cli):
        print(f"Error: llama-cli not found at '{args.llama_cli}'. Please compile it first.")
        sys.exit(1)
        
    if not os.path.exists(args.model):
        print(f"Error: Model not found at '{args.model}'. Please run a demo script to download the model.")
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
    
    run_configs = {
        "Baseline": base_cmd,
        "TurboQuant+": base_cmd + [
            "--cache-type-k", "turbo4",
            "--cache-type-v", "turbo3",
            "--no-turbo-async"
        ],
        "LLMTuning": base_cmd + [
            "--turbo-async"
        ],
        "TurboTuning": base_cmd + [
            "--cache-type-k", "turbo4",
            "--cache-type-v", "turbo3",
            "--turbo-async"
        ]
    }
    
    config_names = list(run_configs.keys())
    all_results = {}
    
    for name in config_names:
        all_results[name] = run_benchmark(name, run_configs[name], env)
        
    print_results_table(all_results, config_names)
    plot_results(all_results, config_names, args.model)

if __name__ == "__main__":
    main()
