#!/bin/bash

# Step 1: 用 LoRA_Adapter_MFA 模型从 CN-Celeb mossformergan 增强数据中提取说话人嵌入
# 处理 CN-Celeb_wav 和 CN-Celeb2_flac 两个数据集
# 支持多 GPU 并行处理（通过文件分片）

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

# ===== 配置 =====
BASE="/root/group-shared/voiceprint/data/speech/speaker_verification/cnceleb_mossformergan_enhanced"
DATA_ROOT="${BASE}"
CHECKPOINT="/root/code/github_repos/LI-MSV-TidyVoice2026-fork/recipes/DeepASV/results/checkpoints/Lora_Adapter_MFA/ckpt_0027_6000item.pth"
TRAIN_YAML="/root/code/github_repos/LI-MSV-TidyVoice2026-fork/recipes/DeepASV/results/checkpoints/Lora_Adapter_MFA/train.yaml"
OUTPUT_DIR="${BASE}/embeddings_w2vbert_lora/embeddings_utterances"
NUM_GPUS=4
PROCS_PER_GPU=8
NUM_WORKERS=8
SKIP_EXISTING=true
RANDOM_SHUFFLE=true
RANDOM_SEED=42
MAX_FILES=0

TOTAL_PROCS=$((NUM_GPUS * PROCS_PER_GPU))
TOTAL_STRIDE=$TOTAL_PROCS

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== CN-Celeb 说话人嵌入提取 (${NUM_GPUS} GPU x ${PROCS_PER_GPU} proc = ${TOTAL_PROCS} total) ===${NC}"
echo -e "  数据集: CN-Celeb_wav + CN-Celeb2_flac"
echo -e "  数据目录: ${DATA_ROOT}"
echo -e "  模型: ${CHECKPOINT}"
echo -e "  配置: ${TRAIN_YAML}"
echo -e "  输出: ${OUTPUT_DIR}"
echo -e "  GPU: ${NUM_GPUS} x ${PROCS_PER_GPU} 进程 (stride=${TOTAL_STRIDE})"
echo -e "  每个进程的 worker: ${NUM_WORKERS}"
echo -e "  跳过已存在: ${SKIP_EXISTING}"

for req in "$CHECKPOINT" "$TRAIN_YAML"; do
    if [ ! -f "$req" ]; then
        echo -e "${RED}错误: 文件不存在: $req${NC}"
        exit 1
    fi
done
if [ ! -d "$DATA_ROOT" ]; then
    echo -e "${RED}错误: 数据目录不存在: $DATA_ROOT${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

PYTHON_SCRIPT="local/extract_embeddings.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误: 脚本不存在: $PYTHON_SCRIPT${NC}"
    exit 1
fi

COMMON_ARGS=(
    --data_root "$DATA_ROOT"
    --checkpoint "$CHECKPOINT"
    --train_yaml "$TRAIN_YAML"
    --output_dir "$OUTPUT_DIR"
    --file_stride "$TOTAL_STRIDE"
    --num_workers "$NUM_WORKERS"
    --random_seed "$RANDOM_SEED"
)
[ "$SKIP_EXISTING" = true ] && COMMON_ARGS+=(--skip_existing)
[ "$RANDOM_SHUFFLE" = true ] && COMMON_ARGS+=(--random_shuffle)
[ "$MAX_FILES" -gt 0 ] && COMMON_ARGS+=(--max_files "$MAX_FILES")

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate asv

echo -e "${GREEN}启动 ${TOTAL_PROCS} 个进程...${NC}"
echo -e "${GREEN}开始时间: $(date)${NC}"
START_TIME=$(date +%s)

PIDS=()
for ((gpu=0; gpu<NUM_GPUS; gpu++)); do
    for ((p=0; p<PROCS_PER_GPU; p++)); do
        offset=$((gpu + p * NUM_GPUS))
        CUDA_VISIBLE_DEVICES=$gpu \
        python3 "$PYTHON_SCRIPT" \
            --device "cuda:0" \
            --file_offset "$offset" \
            "${COMMON_ARGS[@]}" \
            > "/tmp/emb_cnceleb_gpu${gpu}_p${p}.log" 2>&1 &
        PIDS+=($!)
        echo "  GPU${gpu}.proc${p} (offset=${offset}): PID ${PIDS[-1]}"
    done
done

echo "等待所有 ${TOTAL_PROCS} 个进程..."
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo -e "  ${GREEN}[$((i+1))/${TOTAL_PROCS}] 完成${NC}"
    else
        echo -e "  ${RED}[$((i+1))/${TOTAL_PROCS}] 失败${NC}"
        FAILED=1
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo -e "结束时间: $(date), 耗时: ${ELAPSED}s"

if [ -d "$OUTPUT_DIR" ]; then
    extracted=$(find "$OUTPUT_DIR" -name "*.pkl" 2>/dev/null | wc -l)
    echo -e "提取嵌入总数: ${extracted}"
fi

if [ "$FAILED" -eq 1 ]; then
    echo -e "${RED}部分进程失败，查看 /tmp/emb_cnceleb_gpu*_p*.log${NC}"
    exit 1
fi

echo -e "${GREEN}完成。${NC}"
