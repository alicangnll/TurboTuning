"""
TurboQuant+ Prefix KV Cache
============================
Wraps llama-cli to transparently cache and restore the KV state after prefill.

On the first run with a given (model, system_prompt) pair, the KV state is saved
to ~/.cache/turboquant/kv/<hash>.bin after the prompt is processed.
On subsequent runs the saved state is loaded, skipping the prefill entirely.

Usage (standalone):
    python -m turboquant.prefix_cache \\
        --binary llama-cpp-turboquant/build/bin/llama-cli \\
        --model  models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \\
        --system-prompt "You are a helpful assistant." \\
        -- -n 256 -i

Usage (from Python):
    from turboquant.prefix_cache import PrefixCache
    cache = PrefixCache(binary, model, system_prompt)
    cache.run(extra_args=["-n", "256", "-i"])
"""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

_CACHE_DIR = Path.home() / ".cache" / "turboquant" / "kv"
_CACHE_VERSION = "v1"  # bump when state format changes (e.g. after major llama.cpp update)


def _cache_path(model: str, system_prompt: str,
                cache_type_k: str, cache_type_v: str) -> Path:
    """Deterministic path from (model basename, system_prompt, kv types)."""
    key = f"{_CACHE_VERSION}|{os.path.basename(model)}|{system_prompt}|{cache_type_k}|{cache_type_v}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{h}.bin"


class PrefixCache:
    def __init__(self,
                 binary: str,
                 model: str,
                 system_prompt: str = "",
                 cache_type_k: str = "turbo4",
                 cache_type_v: str = "turbo4",
                 ctx: int = 2048,
                 ngl: int = 99,
                 readonly: bool = False):
        self.binary       = binary
        self.model        = model
        self.system_prompt = system_prompt
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.ctx          = ctx
        self.ngl          = ngl
        self.readonly     = readonly
        self._cache_file  = _cache_path(model, system_prompt, cache_type_k, cache_type_v)

    @property
    def cache_file(self) -> Path:
        return self._cache_file

    @property
    def is_cached(self) -> bool:
        return self._cache_file.exists() and self._cache_file.stat().st_size > 0

    def invalidate(self) -> None:
        """Remove the cached state for this (model, prompt) pair."""
        if self._cache_file.exists():
            self._cache_file.unlink()
            print(f"[prefix_cache] Invalidated: {self._cache_file}")

    def run(self, extra_args: Optional[List[str]] = None,
            env: Optional[dict] = None) -> int:
        """Launch llama-cli with prompt-cache wired in. Returns exit code."""
        extra_args = extra_args or []
        cmd = [
            self.binary,
            "-m",   self.model,
            "-ngl", str(self.ngl),
            "-c",   str(self.ctx),
            "--cache-type-k", self.cache_type_k,
            "--cache-type-v", self.cache_type_v,
            "--prompt-cache",    str(self._cache_file),
            "--prompt-cache-all",
        ]
        if self.readonly:
            cmd.append("--prompt-cache-ro")
        if self.system_prompt:
            cmd += ["-sys", self.system_prompt]
        cmd += extra_args

        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        status = "HIT" if self.is_cached else "MISS (will save after prefill)"
        print(f"[prefix_cache] {self._cache_file.name}  [{status}]", flush=True)

        result = subprocess.run(cmd, env=run_env)
        return result.returncode


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="TurboQuant+ prefix KV cache wrapper for llama-cli")
    ap.add_argument("--binary",  required=True)
    ap.add_argument("--model",   required=True)
    ap.add_argument("--system-prompt", default="")
    ap.add_argument("--cache-type-k", default="turbo4")
    ap.add_argument("--cache-type-v", default="turbo4")
    ap.add_argument("--ctx",  type=int, default=2048)
    ap.add_argument("--ngl",  type=int, default=99)
    ap.add_argument("--readonly",  action="store_true",
                    help="Load cache but do not update it")
    ap.add_argument("--invalidate", action="store_true",
                    help="Delete cached state and exit")
    ap.add_argument("--show-path",  action="store_true",
                    help="Print cache file path and exit")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="Extra args forwarded to llama-cli (after --)")
    return ap


def main():
    ap = _build_arg_parser()
    args = ap.parse_args()

    cache = PrefixCache(
        binary        = args.binary,
        model         = args.model,
        system_prompt = args.system_prompt,
        cache_type_k  = args.cache_type_k,
        cache_type_v  = args.cache_type_v,
        ctx           = args.ctx,
        ngl           = args.ngl,
        readonly      = args.readonly,
    )

    if args.show_path:
        print(cache.cache_file)
        return

    if args.invalidate:
        cache.invalidate()
        return

    # Strip leading '--' separator if present
    extra = args.rest
    if extra and extra[0] == "--":
        extra = extra[1:]

    sys.exit(cache.run(extra_args=extra))


if __name__ == "__main__":
    main()
