"""
Mixed-Precision Per-Layer Profiler for TurboQuant+
====================================================
Sweeps TURBO_LAYER_ADAPTIVE modes 0-7 (and optionally a JSON config via mode 9)
against a target prompt, compares output perplexity / token-overlap quality,
then emits a recommended per-layer JSON config for TURBO_LAYER_CONFIG_FILE.

Usage:
    python turboquant/layer_profiler.py \
        --binary llama-cpp-turboquant/build/bin/llama-cli \
        --model  models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
        --prompt "The quick brown fox" \
        --n-predict 64 \
        --modes 0,1,2,5,6,7 \
        --output-json layer_config.json
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

# ── helpers ──────────────────────────────────────────────────────────────────

def _run_llama(binary: str, model: str, prompt: str, n_predict: int,
               extra_env: Dict[str, str], cache_k: str = "turbo4",
               cache_v: str = "turbo2", ctx: int = 512,
               timeout: int = 120) -> Tuple[str, float]:
    """Run llama-cli and return (output_text, wall_seconds)."""
    env = os.environ.copy()
    env.update(extra_env)
    cmd = [
        binary,
        "-m", model,
        "-ngl", "99",
        "-c", str(ctx),
        "--cache-type-k", cache_k,
        "--cache-type-v", cache_v,
        "-n", str(n_predict),
        "-p", prompt,
        "--no-display-prompt",
        "-e",          # escape newlines in output
        "--log-disable",
    ]
    t0 = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                env=env, timeout=timeout)
        elapsed = time.monotonic() - t0
        return result.stdout.strip(), elapsed
    except subprocess.TimeoutExpired:
        return "<TIMEOUT>", timeout


def _token_overlap(ref: str, candidate: str) -> float:
    """Jaccard similarity over whitespace-split tokens (rough quality proxy)."""
    ref_toks = set(ref.lower().split())
    cand_toks = set(candidate.lower().split())
    if not ref_toks and not cand_toks:
        return 1.0
    if not ref_toks or not cand_toks:
        return 0.0
    return len(ref_toks & cand_toks) / len(ref_toks | cand_toks)


def _char_edit_distance_ratio(ref: str, cand: str) -> float:
    """Normalised Levenshtein ratio (0=identical, 1=completely different).
    Uses the fast O(min(m,n)) DP row approach."""
    if ref == cand:
        return 0.0
    m, n = len(ref), len(cand)
    if m == 0 or n == 0:
        return 1.0
    # Keep two rows
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == cand[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, [0] * (n + 1)
    dist = prev[n]
    return dist / max(m, n)


# ── per-layer JSON builder ────────────────────────────────────────────────────

_ADAPTIVE_MODE_LABEL = {
    0: "uniform",
    1: "q8_first_last_4",
    2: "q8_last_8",
    5: "boundary_v_turbo4",
    6: "v_last8_turbo4",
    7: "boundary_v_q8 (recommended)",
}

_TYPE_MAP = {
    "turbo4": "GGML_TYPE_TURBO4_0",
    "turbo3": "GGML_TYPE_TURBO3_0",
    "turbo2": "GGML_TYPE_TURBO2_0",
    "q8_0":   "GGML_TYPE_Q8_0",
    "f16":    "GGML_TYPE_F16",
}


def _build_per_layer_json(mode: int, n_layers: int,
                          base_k: str = "turbo4",
                          base_v: str = "turbo2") -> Dict:
    """Mirror the C++ logic for each mode and return a layer-config dict."""
    k_types: List[str] = []
    v_types: List[str] = []

    for il in range(n_layers):
        lk, lv = base_k, base_v

        if mode == 1 and n_layers >= 8:
            if il < 4 or il >= n_layers - 4:
                lk, lv = "q8_0", "q8_0"

        elif mode == 2 and n_layers >= 8:
            if il >= n_layers - 8:
                lk, lv = "q8_0", "q8_0"

        elif mode == 5 and n_layers >= 8:
            is_boundary = (il < 2 or il >= n_layers - 2)
            lv = "turbo4" if is_boundary else "turbo2"

        elif mode == 6 and n_layers >= 8:
            lv = "turbo4" if il >= n_layers - 8 else "turbo2"

        elif mode == 7 and n_layers >= 8:
            is_boundary = (il < 2 or il >= n_layers - 2)
            lv = "q8_0" if is_boundary else "turbo2"

        k_types.append(lk)
        v_types.append(lv)

    return {"k": k_types, "v": v_types}


# ── main profiler ─────────────────────────────────────────────────────────────

def profile(binary: str, model: str, prompt: str, n_predict: int,
            modes: List[int], n_layers: int,
            cache_k: str, cache_v: str, ctx: int,
            output_json: Optional[str]) -> None:

    print(f"\n{'='*60}")
    print("  TurboQuant+ Mixed-Precision Per-Layer Profiler")
    print(f"  Model  : {model}")
    print(f"  Prompt : {prompt[:60]}{'...' if len(prompt)>60 else ''}")
    print(f"  Modes  : {modes}")
    print(f"{'='*60}\n")

    results = []  # list of (mode, label, output, elapsed, overlap, edit_ratio)

    # Baseline: mode 0 (uniform)
    print("[baseline] Running mode 0 (uniform) ...")
    ref_out, ref_t = _run_llama(binary, model, prompt, n_predict,
                                {"TURBO_LAYER_ADAPTIVE": "0"},
                                cache_k, cache_v, ctx)
    print(f"  -> {ref_t:.1f}s | {len(ref_out.split())} tokens\n")
    results.append((0, "uniform", ref_out, ref_t, 1.0, 0.0))

    for mode in modes:
        if mode == 0:
            continue  # already done as baseline
        label = _ADAPTIVE_MODE_LABEL.get(mode, f"mode_{mode}")
        print(f"[mode {mode}] {label} ...")
        out, elapsed = _run_llama(binary, model, prompt, n_predict,
                                  {"TURBO_LAYER_ADAPTIVE": str(mode)},
                                  cache_k, cache_v, ctx)
        overlap = _token_overlap(ref_out, out)
        edit_r  = _char_edit_distance_ratio(ref_out, out)
        print(f"  -> {elapsed:.1f}s | overlap={overlap:.3f} | edit_dist={edit_r:.3f}")
        results.append((mode, label, out, elapsed, overlap, edit_r))

    # ── ranking ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  RESULTS (ranked by quality → speed)")
    print(f"{'─'*60}")
    # score = 0.7*overlap + 0.3*(1-edit_ratio) — higher is better
    ranked = sorted(results, key=lambda r: -(0.7*r[4] + 0.3*(1-r[5])))
    for rank, (mode, label, _, elapsed, overlap, edit_r) in enumerate(ranked, 1):
        score = 0.7*overlap + 0.3*(1-edit_r)
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  #{rank:2d}  mode={mode:2d}  score={score:.4f}  "
              f"overlap={overlap:.3f}  edit={edit_r:.3f}  "
              f"t={elapsed:.1f}s  [{label}]{marker}")

    best_mode = ranked[0][0]
    print(f"\n  Recommended mode: {best_mode} ({_ADAPTIVE_MODE_LABEL.get(best_mode, '?')})")

    # ── emit JSON ─────────────────────────────────────────────────────────────
    if output_json:
        cfg = _build_per_layer_json(best_mode, n_layers, cache_k, cache_v)
        with open(output_json, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"\n  Per-layer config written to: {output_json}")
        print(f"  Usage: TURBO_LAYER_ADAPTIVE=9 TURBO_LAYER_CONFIG_FILE={output_json}")

    print(f"\n{'='*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_modes(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(
        description="TurboQuant+ mixed-precision per-layer profiler")
    ap.add_argument("--binary",  required=True,
                    help="Path to llama-cli binary")
    ap.add_argument("--model",   required=True,
                    help="Path to GGUF model file")
    ap.add_argument("--prompt",  default="The meaning of life is",
                    help="Prompt string for quality comparison")
    ap.add_argument("--n-predict", type=int, default=64,
                    help="Tokens to generate per run (default 64)")
    ap.add_argument("--modes", default="0,1,2,5,6,7",
                    help="Comma-separated TURBO_LAYER_ADAPTIVE modes to sweep")
    ap.add_argument("--n-layers", type=int, default=32,
                    help="Number of layers in the model (for JSON config output)")
    ap.add_argument("--cache-k", default="turbo4",
                    help="Base K cache type (default: turbo4)")
    ap.add_argument("--cache-v", default="turbo2",
                    help="Base V cache type (default: turbo2)")
    ap.add_argument("--ctx", type=int, default=512,
                    help="Context length (default: 512)")
    ap.add_argument("--output-json", default=None,
                    help="Write recommended per-layer JSON config to this file")
    args = ap.parse_args()

    if not os.path.isfile(args.binary):
        print(f"ERROR: binary not found: {args.binary}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"ERROR: model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    profile(
        binary=args.binary,
        model=args.model,
        prompt=args.prompt,
        n_predict=args.n_predict,
        modes=_parse_modes(args.modes),
        n_layers=args.n_layers,
        cache_k=args.cache_k,
        cache_v=args.cache_v,
        ctx=args.ctx,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
