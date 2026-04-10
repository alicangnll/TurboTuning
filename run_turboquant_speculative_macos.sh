#!/bin/bash
# TurboQuant+ Speculative Decoding Demo (macOS)
# -----------------------------------------------
# Target model: Llama 3.1 8B Instruct  (turbo4 KV)
# Draft  model: Qwen 2.5 0.5B Instruct (turbo4 KV)
#
# Speculative decoding generates draft tokens from the small model
# and verifies them in parallel with the large model — yielding
# 1.5-3x wall-clock speedup at identical output quality.
#
# Usage: ./run_turboquant_speculative_macos.sh

set -e
ROOT_DIR=$(pwd)
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

echo "======== TURBOQUANT+ SPECULATIVE DECODING DEMO ========"

# ── 1. libomp ──────────────────────────────────────────────
install_libomp() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! brew list libomp &>/dev/null; then
            echo ">>> Installing libomp via Homebrew..."
            brew install libomp
        fi
        LIBOMP_PREFIX=$(brew --prefix libomp)
        OMP_ENABLED="ON"
        OMP_C_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP_PREFIX/include"
        OMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I$LIBOMP_PREFIX/include"
        OMP_LIB_PATH="$LIBOMP_PREFIX/lib/libomp.dylib"
    else
        OMP_ENABLED="OFF"
    fi
}
install_libomp

# ── 2. Build ───────────────────────────────────────────────
cd llama-cpp-turboquant

if [[ "$OSTYPE" == "darwin"* ]] && [ ! -d "build" ]; then
    echo ">>> Compiling C++ engine (Metal + OpenMP)..."
    cmake -B build \
        -DGGML_METAL=ON \
        -DGGML_METAL_EMBED_LIBRARY=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CPU_REPACK=ON \
        -DGGML_OPENMP="${OMP_ENABLED:-OFF}" \
        -DOpenMP_C_FLAGS="${OMP_C_FLAGS:-}" \
        -DOpenMP_C_LIB_NAMES="omp" \
        -DOpenMP_CXX_FLAGS="${OMP_CXX_FLAGS:-}" \
        -DOpenMP_CXX_LIB_NAMES="omp" \
        -DOpenMP_omp_LIBRARY="${OMP_LIB_PATH:-}"
    cmake --build build -j --target llama-speculative
elif [ ! -d "build" ]; then
    echo ">>> Compiling C++ engine..."
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build -j --target llama-speculative
fi

LLAMA_SPEC="./build/bin/llama-speculative"
if [ ! -f "$LLAMA_SPEC" ]; then
    echo "ERROR: llama-speculative binary not found at $LLAMA_SPEC"
    echo "       Make sure the build includes the speculative target."
    exit 1
fi

# ── 3. Model downloads ─────────────────────────────────────
TARGET_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
TARGET_FILE="../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

DRAFT_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
DRAFT_FILE="../models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

mkdir -p ../models

if [ ! -f "$TARGET_FILE" ]; then
    echo ">>> Downloading target model: Llama 3.1 8B..."
    curl -L -o "$TARGET_FILE" "$TARGET_URL"
fi
if [ ! -f "$DRAFT_FILE" ]; then
    echo ">>> Downloading draft model: Qwen 0.5B..."
    curl -L -o "$DRAFT_FILE" "$DRAFT_URL"
fi

# ── 4. Settings ────────────────────────────────────────────
THREADS=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
SYSTEM_PROMPT="You are a helpful AI assistant. Be concise."

# Number of draft tokens to speculatively generate before verification.
# Higher = more speedup potential, but more wasted compute on mismatches.
DRAFT_N=8

# TurboQuant+ KV cache types — applies to BOTH models.
CACHE_K="turbo4"
CACHE_V="turbo4"

echo ""
echo ">>> Target : Llama 3.1 8B Instruct  (KV: $CACHE_K/$CACHE_V)"
echo ">>> Draft  : Qwen 2.5 0.5B Instruct (KV: $CACHE_K/$CACHE_V)"
echo ">>> Draft N: $DRAFT_N tokens ahead"
echo ">>> Threads: $THREADS"
echo ""

# ── 5. Run ─────────────────────────────────────────────────
echo ">>> Starting speculative decoding session..."
echo "    (Ctrl-C to exit)"
echo ""

"$LLAMA_SPEC" \
    -m "$TARGET_FILE" \
    -md "$DRAFT_FILE" \
    -ngl 99 \
    -ngld 99 \
    -t "$THREADS" \
    -c 2048 \
    --cache-type-k  "$CACHE_K" \
    --cache-type-v  "$CACHE_V" \
    --cache-type-k-draft "$CACHE_K" \
    --cache-type-v-draft "$CACHE_V" \
    -n "$DRAFT_N" \
    --draft "$DRAFT_N" \
    -sys "$SYSTEM_PROMPT" \
    -i \
    --chat-template llama3 \
    --color

echo ""
echo "======== Session ended ========"
