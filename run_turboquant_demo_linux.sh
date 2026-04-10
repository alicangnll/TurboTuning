#!/bin/bash

# ============================================================
# TURBOQUANT+ LINUX DEMO SCRIPT
# Optimized for CUDA (NVIDIA), ROCm (AMD), and OpenMP (CPU)
# ============================================================

set -e

ROOT_DIR=$(pwd)

echo "=============== TURBOQUANT LINUX DEMO ==============="

SYSTEM_PROMPT="You are a Technical Research AI. Respond in the user's language when they write in that language."

# Helper: Detect Backend
detect_backend() {
    echo ">>> Detect hardware and backend..."
    if command -v nvidia-smi &>/dev/null; then
        echo ">>> NVIDIA GPU Detected. Using CUDA backend."
        CMAKE_FLAGS="-DGGML_CUDA=ON"
    elif command -v rocm-smi &>/dev/null || lsmod | grep -q amdgpu; then
        echo ">>> AMD GPU Detected. Using ROCm/HIP backend."
        CMAKE_FLAGS="-DGGML_HIPBLAS=ON"
    else
        echo ">>> No compatible GPU found. Defaulting to CPU (OpenMP)."
        CMAKE_FLAGS="-DGGML_OPENMP=ON"
    fi
}

install_linux_deps() {
    echo ">>> Checking for dependencies..."
    if command -v apt-get &>/dev/null; then
        echo ">>> Linux (Debian/Ubuntu) detected."
        sudo apt-get update && sudo apt-get install -y libomp-dev cmake build-essential curl git
    elif command -v pacman &>/dev/null; then
        echo ">>> Linux (Arch) detected."
        sudo pacman -S --noconfirm libomp cmake base-devel curl git
    elif command -v dnf &>/dev/null; then
        echo ">>> Linux (Fedora) detected."
        sudo dnf install -y libomp-devel cmake make gcc-c++ curl git
    fi
}

# --- MAIN EXECUTION ---
detect_backend
install_linux_deps

# Part 1: Compile
echo ">>> [1/3] Compiling C++ engine..."
cd llama-cpp-turboquant

echo ">>> Compiling with flags: $CMAKE_FLAGS"
cmake -B build -DCMAKE_BUILD_TYPE=Release $CMAKE_FLAGS
cmake --build build -j --target llama-cli

# Part 2: Model Selection
echo ">>> [2/3] Select the model you want to run:"
echo "1) Llama 3.1 8B Instruct (~5 GB)"
echo "2) Qwen 2.5 32B Instruct (~20 GB)"
echo "3) Command R+ 104B (~43 GB)"
echo "4) Qwen 2.5 0.5B Instruct (~400 MB)"
echo "5) Llama-3-405B / 500B Class"
echo "6) GPT 20B (OpenAI OSS-20B Class - ~12 GB)"
echo "8) Qwen 2.5 Coder 7B/8B (~5 GB)"
read -p "Your choice (1/2/3/4/5/6/8) [Default: 4]: " model_choice
model_choice=${model_choice:-4}

mkdir -p $ROOT_DIR/models

case "$model_choice" in
  1|"8B")
    MODEL_NAME="Llama 3.1 8B"
    MODEL_URL="https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    MODEL_FILE="$ROOT_DIR/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    CHAT_TEMPLATE="llama3"
    ;;
  2|"32B")
    MODEL_NAME="Qwen 2.5 32B"
    MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
    MODEL_FILE="$ROOT_DIR/models/Qwen2.5-32B-Instruct-Q4_K_M.gguf"
    CHAT_TEMPLATE="chatml"
    ;;
  3|"100B")
    MODEL_NAME="Command R+ 104B"
    MODEL_URL="https://huggingface.co/mradermacher/c4ai-command-r-plus-08-2024-GGUF/resolve/main/c4ai-command-r-plus-08-2024.Q2_K.gguf"
    MODEL_FILE="$ROOT_DIR/models/c4ai-command-r-plus-08-2024.Q2_K.gguf"
    CHAT_TEMPLATE="command-r"
    ;;
  5|"405B")
    MODEL_NAME="Llama 3.1 405B"
    MODEL_URL="https://huggingface.co/mradermacher/Meta-Llama-3.1-405B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-405B-Instruct.Q2_K.gguf"
    MODEL_FILE="$ROOT_DIR/models/Meta-Llama-3.1-405B-Instruct.Q2_K.gguf"
    CHAT_TEMPLATE="llama3"
    ;;
  6|"20B")
    MODEL_NAME="GPT 20B (OSS)"
    MODEL_URL="https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF/resolve/main/openai_gpt-oss-20b-Q4_K_M.gguf"
    MODEL_FILE="$ROOT_DIR/models/openai_gpt-oss-20b-Q4_K_M.gguf"
    CHAT_TEMPLATE="none"
    ;;
  8|"8BC"|"8bc"|"7B"|"7b")
    MODEL_NAME="Qwen 2.5 Coder 7B"
    MODEL_URL="https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    MODEL_FILE="$ROOT_DIR/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"
    CHAT_TEMPLATE="chatml"
    ;;
  *)
    MODEL_NAME="Qwen 2.5 0.5B"
    MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    MODEL_FILE="$ROOT_DIR/models/qwen2.5-0.5b-q4_k_m.gguf"
    CHAT_TEMPLATE="chatml"
    ;;
esac

if [ ! -f "$MODEL_FILE" ]; then
    echo ">>> Downloading $MODEL_NAME..."
    curl -L -o "$MODEL_FILE" "$MODEL_URL"
fi
MODEL_PATH="$MODEL_FILE"

# Part 3: Launch with LLMTuning
echo ">>> [3/3] Starting $MODEL_NAME with TurboQuant+ & LLMTuning..."
echo ">>> LLMTuning: ON (auto-activated via TurboQuant cache types)"
echo ">>> Cache: turbo4/turbo3 | Context: 512 | NGL: 99"

THREADS=$(nproc 2>/dev/null || echo 4)

echo ""
echo ">>> Starting interactive session..."
echo ""

./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -t $THREADS \
    -c 512 \
    --cache-type-k turbo4 \
    --cache-type-v turbo3 \
    -sys "$SYSTEM_PROMPT" \
    -p "Hello! What is 2+2?" \
    -n 300 \
    --turbo-async \
    --no-display-prompt

echo "-----------------------------------------------"
echo ">>> Demo completed!"
