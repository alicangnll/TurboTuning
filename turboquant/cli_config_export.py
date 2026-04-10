import json
import argparse
import sys

# Documented in docs/memory-rss-targets.md — primary tuning goal is peak process RSS.
_MEM_TIER_PROFILE = {
    "1": {
        "batch_size": 512,
        "ubatch_size": 256,
        "rss_target_note": "Performance: higher peak RSS acceptable; prioritize throughput.",
    },
    "2": {
        "batch_size": 256,
        "ubatch_size": 128,
        "rss_target_note": "Balanced: moderate peak RSS (~24GB-class systems).",
    },
    "3": {
        "batch_size": 32,
        "ubatch_size": 32,
        "rss_target_note": "Ultra-Eco: minimize peak RSS (~16GB-class systems).",
    },
}

def _apply_mem_tier(config, mem_choice):
    tier = _MEM_TIER_PROFILE.get(mem_choice, _MEM_TIER_PROFILE["2"])
    config["batch_size"] = tier["batch_size"]
    config["ubatch_size"] = tier["ubatch_size"]
    config["rss_target_note"] = tier["rss_target_note"]
    config["primary_ram_metric"] = "peak_rss"
    config["lossless_definition"] = (
        "product_default: stable chat on same GGUF quant; turbo KV is approximate vs f16 KV"
    )
    config["mem_tier"] = mem_choice
    return config

def get_optimal_config(model_choice, mem_choice):
    # Default values - LLMTuning Universal Minimization
    config = {
        "ctx_len": 2048,
        "cache_type_k": "turbo4", # Standardize on high-compression
        "cache_type_v": "turbo4",
        "num_layers": 32,
        "num_heads": 32,
        "head_dim": 128,
        "chat_template": "",
        "extra_args": "",
        # Speculative decoding (disabled by default; set draft_model to enable)
        "speculative": {
            "enabled": False,
            "draft_model_url": None,
            "draft_model_file": None,
            "draft_cache_type_k": "turbo4",
            "draft_cache_type_v": "turbo4",
            "draft_n": 8,          # tokens to speculatively generate ahead
            "ngld": 99,            # draft model GPU layers
        }
    }

    # Model specific architectures (LLMTuning Intelligence).
    # KV cache: TurboQuant types only (turbo4 / turbo2) — no q8_0 fallback in policy.
    if model_choice in ["1", "8B", "8b"]:  # Llama 3.1 8B
        config.update({
            "num_layers": 32,
            "num_heads": 32,
            "head_dim": 128,
            "ctx_len": 4096 if mem_choice == "1" else (2048 if mem_choice == "2" else 1024),
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo2" if mem_choice == "3" else "turbo4",
            "chat_template": "llama3",
            "extra_args": "-fa on",
            "speculative": {
                "enabled": True,
                "draft_model_url": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                "draft_model_file": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
                "draft_cache_type_k": "turbo4",
                "draft_cache_type_v": "turbo4",
                "draft_n": 8,
                "ngld": 99,
            },
        })
 
    elif model_choice in ["8", "8BC", "8bc", "7B", "7b"]:  # Qwen 2.5 Coder 7B / 8B
        config.update({
            "num_layers": 28,
            "num_heads": 28,
            "head_dim": 128,
            "ctx_len": 4096 if mem_choice == "1" else (2048 if mem_choice == "2" else 1024),
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo2" if mem_choice == "3" else "turbo4",
            "chat_template": "qwen2",
            "extra_args": "-fa on",
        })

    elif model_choice in ["4", "0.5B", "0.5b"]:  # Qwen 2.5 0.5B Instruct
        config.update({
            "num_layers": 24,
            "num_heads": 14,
            "head_dim": 64,
            "ctx_len": 2048 if mem_choice == "1" else (1024 if mem_choice == "2" else 512),
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo2" if mem_choice == "3" else "turbo4",
            "chat_template": "qwen2",
        })

    elif model_choice in ["5", "405B", "405b"]:  # Llama 3.1 405B class
        config.update({
            "num_layers": 126,
            "num_heads": 128,
            "head_dim": 128,
            "ctx_len": 512 if mem_choice == "1" else (256 if mem_choice == "2" else 128),
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo2" if mem_choice == "3" else "turbo4",
            "chat_template": "llama3",
        })

    elif model_choice in ["2", "32B", "32b"]: # Qwen 2.5 32B
        config.update({
            "num_layers": 64,
            "num_heads": 40,
            "head_dim": 128,
            "ctx_len": 1024 if mem_choice == "1" else (512 if mem_choice == "2" else 256),
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo2" if mem_choice == "3" else "turbo4", 
            "chat_template": "qwen2"
        })

    elif model_choice in ["3", "100B", "100b"]: # Command R+ 104B
        config.update({
            "num_layers": 96,
            "num_heads": 128,
            "head_dim": 128,
            "ctx_len": 1024 if mem_choice == "1" else (512 if mem_choice == "2" else 256),
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo2" if mem_choice == "3" else "turbo4",
            "chat_template": "command-r"
        })

    elif model_choice in ["6", "20B", "20b"]: # GPT 20B OSS (MoE)
        config.update({
            "num_layers": 24,
            "num_heads": 8,
            "head_dim": 64,
            "ctx_len": 2048 if mem_choice == "1" else 1024,
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo4",
            "chat_template": "none",
            "extra_args": "-fa on",
        })

    elif model_choice in ["7", "31B", "31b"]: # Gemma 4 31B
        config.update({
            "num_layers": 48,
            "num_heads": 32,
            "head_dim": 128,
            "ctx_len": 4096 if mem_choice == "1" else (2048 if mem_choice == "2" else 512),
            "cache_type_k": "turbo4",
            "cache_type_v": "turbo2" if mem_choice == "3" else "turbo4",
            "chat_template": "gemma",
            "extra_args": "-fa on"
        })

    return _apply_mem_tier(config, mem_choice)


def _emit_bat_env(config):
    """Windows cmd.exe: SET lines for run_turboquant_demo.bat (quoted values)."""

    def esc(s):
        return str(s).replace('"', '""')

    pairs = [
        ("CTX", config["ctx_len"]),
        ("CACHE_TYPE_K", config["cache_type_k"]),
        ("CACHE_TYPE_V", config["cache_type_v"]),
        ("BATCH_SIZE", config["batch_size"]),
        ("UBATCH_SIZE", config["ubatch_size"]),
        ("NUM_LAYERS", config["num_layers"]),
        ("CHAT_TEMPLATE", config.get("chat_template") or ""),
        ("EXTRA_POLICY", (config.get("extra_args") or "").strip()),
    ]
    for name, val in pairs:
        print(f'set "{name}={esc(val)}"')

    spec = config.get("speculative", {})
    if spec.get("enabled"):
        spec_pairs = [
            ("SPEC_DRAFT_MODEL_URL",  spec.get("draft_model_url", "")),
            ("SPEC_DRAFT_MODEL_FILE", spec.get("draft_model_file", "")),
            ("SPEC_CACHE_K_DRAFT",    spec.get("draft_cache_type_k", "turbo4")),
            ("SPEC_CACHE_V_DRAFT",    spec.get("draft_cache_type_v", "turbo4")),
            ("SPEC_DRAFT_N",          spec.get("draft_n", 8)),
            ("SPEC_NGLD",             spec.get("ngld", 99)),
        ]
        for name, val in spec_pairs:
            print(f'set "{name}={esc(val)}"')


def _emit_speculative_sh(config):
    """Print a ready-to-run shell snippet for speculative decoding."""
    spec = config.get("speculative", {})
    if not spec.get("enabled"):
        print("# Speculative decoding not configured for this model choice.")
        return
    lines = [
        "# TurboQuant+ Speculative Decoding — generated by cli_config_export.py",
        f"# Draft model : {spec.get('draft_model_file', 'N/A')}",
        f"# Draft N     : {spec.get('draft_n', 8)}",
        "",
        f"DRAFT_FILE=\"{spec.get('draft_model_file', '')}\"\n"
        f"TARGET_FILE=\"<your_target_model.gguf>\"",
        "",
        "./build/bin/llama-speculative \\",
        f"    -m \"$TARGET_FILE\" \\",
        f"    -md \"$DRAFT_FILE\" \\",
        f"    -ngl 99 \\",
        f"    -ngld {spec.get('ngld', 99)} \\",
        f"    --cache-type-k {config.get('cache_type_k', 'turbo4')} \\",
        f"    --cache-type-v {config.get('cache_type_v', 'turbo4')} \\",
        f"    --cache-type-k-draft {spec.get('draft_cache_type_k', 'turbo4')} \\",
        f"    --cache-type-v-draft {spec.get('draft_cache_type_v', 'turbo4')} \\",
        f"    --draft {spec.get('draft_n', 8)} \\",
        f"    -c {config.get('ctx_len', 2048)} \\",
        f"    -i --chat-template {config.get('chat_template', 'llama3')} --color",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-choice", required=True)
    parser.add_argument("--mem-choice", required=True)
    parser.add_argument(
        "--emit",
        choices=["json", "bat", "speculative-sh"],
        default="json",
        help="json: full policy to stdout; bat: SET lines for Windows demo; "
             "speculative-sh: shell snippet for speculative decoding",
    )
    args = parser.parse_args()

    config = get_optimal_config(args.model_choice, args.mem_choice)
    if args.emit == "bat":
        _emit_bat_env(config)
    elif args.emit == "speculative-sh":
        _emit_speculative_sh(config)
    else:
        print(json.dumps(config))
