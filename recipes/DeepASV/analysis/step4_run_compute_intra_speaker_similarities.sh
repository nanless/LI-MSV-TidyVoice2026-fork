#!/bin/bash

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

BASE="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130"
UTTERANCES_DIR="${BASE}/embeddings_w2vbert_lora_alldatasets/embeddings_utterances"
OUTPUT_DIR="${BASE}/embeddings_w2vbert_lora_alldatasets/intra_speaker_similarities"
MAX_UTTERANCES=1000
NUM_PROCESSES=$(nproc)
CHUNK_SIZE=10
SKIP_EXISTING=true

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== 说话人内两两相似度计算 ===${NC}"
echo -e "  语句嵌入目录: ${UTTERANCES_DIR}"
echo -e "  输出目录: ${OUTPUT_DIR}"
echo -e "  最大语句数/说话人: ${MAX_UTTERANCES}"
echo -e "  进程数: ${NUM_PROCESSES}"
echo -e "  跳过已存在: ${SKIP_EXISTING}"

if [ ! -d "$UTTERANCES_DIR" ]; then
    echo -e "${RED}错误: 语句嵌入目录不存在: $UTTERANCES_DIR${NC}"
    echo -e "${YELLOW}  提示: 先运行 extract_embeddings.py 提取语句嵌入${NC}"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

PYTHON_SCRIPT="local/compute_intra_speaker_similarities.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误: 脚本不存在: $PYTHON_SCRIPT${NC}"
    exit 1
fi

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate asv

echo -e "${GREEN}开始计算...${NC}"
echo -e "${GREEN}开始时间: $(date)${NC}"
START_TIME=$(date +%s)

CMD_ARGS=(
    --utterances_dir "$UTTERANCES_DIR"
    --output_dir "$OUTPUT_DIR"
    --max_utterances "$MAX_UTTERANCES"
    --num_processes "$NUM_PROCESSES"
    --chunk_size "$CHUNK_SIZE"
)

[ "$SKIP_EXISTING" = true ] && CMD_ARGS+=(--skip_existing)

python3 "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo -e "结束时间: $(date), 耗时: ${ELAPSED}s"

echo -e "${GREEN}完成。输出目录: ${OUTPUT_DIR}${NC}"
