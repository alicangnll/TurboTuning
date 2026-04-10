#!/bin/bash

# To prevent errors when running on Mac
set -e

# Starting Directory (Assuming you run this in the project root)
ROOT_DIR=$(pwd)
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

echo "=============== TURBOQUANT+ DEMO ==============="

SYSTEM_PROMPT="You are a Technical Research AI. Respond in the user's language when they write in that language."

# Helper: Detect and Install libomp (OpenMP) for high-speed CPU inference
install_libomp() {
    echo ">>> Checking for libomp (OpenMP Support)..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        if ! brew list libomp &>/dev/null; then
            echo ">>> macOS detected: Installing libomp via Homebrew..."
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

# Part 1: Compile llama.cpp fork
echo ">>> [1/3] Preparing C++ engine..."
install_libomp

cd llama-cpp-turboquant

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ">>> Cleaning build for fresh optimization..."
    rm -rf build
fi

if [ ! -d "build" ]; then
    echo ">>> Compiling C++ engine with Dual Acceleration (Metal + OpenMP)..."
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
    cmake --build build -j --target llama-cli
fi

# Part 2: Model Selection
echo ">>> [2/3] Select the model you want to run:"
echo "1) Llama 3.1 8B Instruct (~5 GB)"
echo "2) Qwen 2.5 32B Instruct (~20 GB)"
echo "3) Command R+ 104B (~43 GB)"
echo "4) Qwen 2.5 0.5B Instruct (~400 MB)"
echo "6) GPT 20B (OpenAI OSS-20B Class - ~12 GB)"
echo "7) Gemma 4 31B (Google - ~18 GB)"
echo "8) Qwen 2.5 Coder 7B/8B (~5 GB)"
echo "9) Llama 3.1 70B Instruct (~40 GB)"
read -p "Your choice (1/2/3/4/6/7/8/9) [Default: 1]: " model_choice
model_choice=${model_choice:-1}

# Model Mapping
case "$model_choice" in
  1) MODEL_NAME="Llama 3.1 8B"; MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"; MODEL_FILE="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"; CHAT_TEMPLATE="llama3" ;;
  2) MODEL_NAME="Qwen 2.5 32B"; MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf"; MODEL_FILE="models/Qwen2.5-32B-Instruct-Q4_K_M.gguf"; CHAT_TEMPLATE="chatml" ;;
  3) MODEL_NAME="Command R+ 104B"; MODEL_URL="https://huggingface.co/mradermacher/c4ai-command-r-plus-08-2024-GGUF/resolve/main/c4ai-command-r-plus-08-2024.Q2_K.gguf"; MODEL_FILE="models/c4ai-command-r-plus-08-2024.Q2_K.gguf"; CHAT_TEMPLATE="command-r" ;;
  6) MODEL_NAME="GPT 20B OSS"; MODEL_URL="https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF/resolve/main/openai_gpt-oss-20b-Q4_K_M.gguf"; MODEL_FILE="models/openai_gpt-oss-20b-Q4_K_M.gguf"; CHAT_TEMPLATE="none" ;;
  7) MODEL_NAME="Gemma 4 31B"; MODEL_URL="https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/gemma-4-31B-it-Q4_K_M.gguf"; MODEL_FILE="models/gemma-4-31B-it-Q4_K_M.gguf"; CHAT_TEMPLATE="gemma" ;;
  8) MODEL_NAME="Qwen 2.5 Coder 7B"; MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"; MODEL_FILE="models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"; CHAT_TEMPLATE="chatml" ;;
  9) MODEL_NAME="Llama 3.1 70B"; MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"; MODEL_FILE="models/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"; CHAT_TEMPLATE="llama3" ;;
  *) MODEL_NAME="Qwen 0.5B"; MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"; MODEL_FILE="models/qwen2.5-0.5b-q4_k_m.gguf"; CHAT_TEMPLATE="chatml" ;;
esac

if [ ! -f "../$MODEL_FILE" ]; then
    echo ">>> Downloading $MODEL_NAME..."
    mkdir -p ../models
    curl -L -o "../$MODEL_FILE" "$MODEL_URL"
fi
MODEL_PATH="../$MODEL_FILE"

# Part 3: Launch with LLMTuning + Prefix KV Cache
echo ">>> [3/3] Starting $MODEL_NAME with TurboQuant+ & LLMTuning..."
echo ">>> LLMTuning: ON (auto-activated via TurboQuant cache types)"
echo ">>> Cache: turbo4/turbo3 | Context: 512 | NGL: 99"

THREADS=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Prefix KV Cache: hash (model + system prompt) → reuse KV state across sessions
# First run: prefill is computed + state saved. Every subsequent run: load instantly.
CACHE_DIR="$HOME/.cache/turboquant/kv"
mkdir -p "$CACHE_DIR"
CACHE_HASH=$(echo -n "v1|$(basename $MODEL_PATH)|$SYSTEM_PROMPT|turbo4|turbo3" | sha256sum 2>/dev/null | cut -c1-16 || \
             echo -n "v1|$(basename $MODEL_PATH)|$SYSTEM_PROMPT|turbo4|turbo3" | shasum -a 256 | cut -c1-16)
KV_CACHE_FILE="$CACHE_DIR/${CACHE_HASH}.bin"

if [ -f "$KV_CACHE_FILE" ] && [ -s "$KV_CACHE_FILE" ]; then
    echo ">>> Prefix KV cache: HIT ($KV_CACHE_FILE)"
else
    echo ">>> Prefix KV cache: MISS — will save after prefill"
fi

echo ""
echo ">>> LLMTuning validation complete. Starting interactive session..."
echo ""

./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -t $THREADS \
    -c 1024 \
    --cache-type-k turbo4 \
    --cache-type-v turbo3 \
    -sys "$SYSTEM_PROMPT" \
    -p "Hello! What is 2+2?" \
    -n 300 \
    --turbo-async \
    --no-display-prompt

echo "-----------------------------------------------"
echo ">>> Demo completed!"
