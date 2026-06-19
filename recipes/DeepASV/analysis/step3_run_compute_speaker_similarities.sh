#!/bin/bash

# Step 3: 计算说话人之间的余弦相似度

set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

# ===== 配置 =====
BASE="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130"
EMBEDDINGS_DIR="${BASE}/embeddings_w2vbert_lora_alldatasets"
UTTERANCES_SUBDIR="embeddings_utterances"
SPEAKERS_SUBDIR="embeddings_speakers"
SIMILARITIES_SUBDIR="speaker_similarity_analysis"
NUM_WORKERS=32
BATCH_SIZE=100
TOP_K=100
SKIP_SIMILARITY=false
MAX_SPEAKERS=0
EXCLUDE_CLONE_PATTERN=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== 说话人相似度计算 ===${NC}"
echo -e "  嵌入目录: ${EMBEDDINGS_DIR}"
echo -e "  语句子目录: ${UTTERANCES_SUBDIR}"
echo -e "  说话人子目录: ${SPEAKERS_SUBDIR}"
echo -e "  相似度输出: ${SIMILARITIES_SUBDIR}"
echo -e "  进程数: ${NUM_WORKERS}"
echo -e "  Batch 大小: ${BATCH_SIZE}"
echo -e "  Top-K: ${TOP_K}"
echo -e "  最大说话人数: ${MAX_SPEAKERS:-全部}"

SPEAKERS_FULL_PATH="$EMBEDDINGS_DIR/$SPEAKERS_SUBDIR"
SIMILARITIES_FULL_PATH="$EMBEDDINGS_DIR/$SIMILARITIES_SUBDIR"

if [ ! -d "$SPEAKERS_FULL_PATH" ]; then
    echo -e "${RED}错误: 说话人嵌入目录不存在: $SPEAKERS_FULL_PATH${NC}"
    echo -e "${YELLOW}  提示: 先运行 step2 计算说话人嵌入${NC}"
    exit 1
fi

PYTHON_SCRIPT="local/compute_speaker_similarities.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo -e "${RED}错误: 脚本不存在: $PYTHON_SCRIPT${NC}"
    exit 1
fi

echo -e "CPU 核心数: $(nproc)"
echo -e "内存: $(free -h | grep '^Mem:' | awk '{print $2}')"

if [ -d "$SPEAKERS_FULL_PATH" ]; then
    total_spks=$(find "$SPEAKERS_FULL_PATH" -name "*.pkl" 2>/dev/null | wc -l)
    echo -e "说话人嵌入总数: ${total_spks}"
fi

source "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate asv 2>/dev/null || true

echo -e "${GREEN}开始计算相似度...${NC}"
echo -e "${GREEN}开始时间: $(date)${NC}"
START_TIME=$(date +%s)

CMD_ARGS=(
    --embeddings_dir "$EMBEDDINGS_DIR"
    --speakers_subdir "$SPEAKERS_SUBDIR"
    --similarities_output_subdir "$SIMILARITIES_SUBDIR"
    --num_workers "$NUM_WORKERS"
    --batch_size "$BATCH_SIZE"
    --top_k "$TOP_K"
)

[ "$SKIP_SIMILARITY" = true ] && CMD_ARGS+=(--skip_similarity)
[ "$MAX_SPEAKERS" -gt 0 ] && CMD_ARGS+=(--max_speakers "$MAX_SPEAKERS")
[ -n "$EXCLUDE_CLONE_PATTERN" ] && CMD_ARGS+=(--exclude_filename_pattern "$EXCLUDE_CLONE_PATTERN")

export PYTHONIOENCODING=UTF-8
python3 "$PYTHON_SCRIPT" "${CMD_ARGS[@]}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo -e "结束时间: $(date), 耗时: ${ELAPSED}s"

echo -e "${YELLOW}输出文件:${NC}"
for f in speaker_keys_mapping.json speaker_top_similarities.json analysis_summary.json extreme_similarity_pairs.json threshold_statistics.json; do
    if [ -f "$SIMILARITIES_FULL_PATH/$f" ]; then
        echo -e "  ${GREEN}[OK]${NC} $SIMILARITIES_FULL_PATH/$f"
    fi
done

echo -e "${GREEN}完成。结果保存在: ${SIMILARITIES_FULL_PATH}${NC}"
