#!/bin/bash

# Step 2: 通过平均所有语句嵌入计算说话人级别的嵌入

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

# ===== 配置 =====
BASE="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130"
UTTERANCES_DIR="${BASE}/embeddings_w2vbert_lora_alldatasets/embeddings_utterances"
SPEAKERS_DIR="${BASE}/embeddings_w2vbert_lora_alldatasets/embeddings_speakers"
MIN_UTTERANCES=1
NUM_PROCESSES=$(nproc)
CHUNK_SIZE=10
SKIP_EXISTING=true
EXCLUDE_VOICEPRINT_PREFIX=""
EXCLUDE_CLONE_PATTERN=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== 说话人嵌入计算（多进程） ===${NC}"
echo -e "  语句嵌入目录: ${UTTERANCES_DIR}"
echo -e "  说话人嵌入输出: ${SPEAKERS_DIR}"
echo -e "  最少语句数: ${MIN_UTTERANCES}"
echo -e "  进程数: ${NUM_PROCESSES}"
echo -e "  Chunk 大小: ${CHUNK_SIZE}"
echo -e "  跳过已存在: ${SKIP_EXISTING}"

if [ ! -d "$UTTERANCES_DIR" ]; then
    echo -e "${RED}错误: 语句嵌入目录不存在: $UTTERANCES_DIR${NC}"
    echo -e "${YELLOW}  提示: 先运行 step1 提取语句嵌入${NC}"
    exit 1
fi

mkdir -p "$SPEAKERS_DIR"

PYTHON_SCRIPT="local/compute_speaker_embeddings.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误: 脚本不存在: $PYTHON_SCRIPT${NC}"
    exit 1
fi

echo -e "CPU 核心数: $(nproc)"
echo -e "内存: $(free -h | grep '^Mem:' | awk '{print $2}')"

if [ -d "$UTTERANCES_DIR" ]; then
    total_utts=$(find "$UTTERANCES_DIR" -name "*.pkl" 2>/dev/null | wc -l)
    echo -e "  语句 pkl 文件总数: ${total_utts}"
fi

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate asv 2>/dev/null || true

echo -e "${GREEN}开始计算说话人嵌入...${NC}"
echo -e "${GREEN}开始时间: $(date)${NC}"
START_TIME=$(date +%s)

CMD_ARGS=(
    --utterances_dir "$UTTERANCES_DIR"
    --speakers_dir "$SPEAKERS_DIR"
    --min_utterances "$MIN_UTTERANCES"
    --num_processes "$NUM_PROCESSES"
    --chunk_size "$CHUNK_SIZE"
)

[ "$SKIP_EXISTING" = true ] && CMD_ARGS+=(--skip_existing)
[ -n "$EXCLUDE_VOICEPRINT_PREFIX" ] && CMD_ARGS+=(--exclude_filename_prefix "$EXCLUDE_VOICEPRINT_PREFIX")
[ -n "$EXCLUDE_CLONE_PATTERN" ] && CMD_ARGS+=(--exclude_filename_pattern "$EXCLUDE_CLONE_PATTERN")

python3 "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo -e "结束时间: $(date), 耗时: ${ELAPSED}s"

if [ -d "$SPEAKERS_DIR" ]; then
    final_spks=$(find "$SPEAKERS_DIR" -name "*.pkl" 2>/dev/null | wc -l)
    echo -e "说话人嵌入总数: ${final_spks}"
fi

echo -e "${GREEN}完成。${NC}"
