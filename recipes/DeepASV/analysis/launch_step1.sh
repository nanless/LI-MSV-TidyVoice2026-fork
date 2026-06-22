#!/bin/bash
set -e
cd /root/code/github_repos/LI-MSV-TidyVoice2026-fork/recipes/DeepASV/analysis
source /root/miniforge3/etc/profile.d/conda.sh && conda activate asv

PY=/root/miniforge3/envs/asv/bin/python3
BASE="/root/group-shared/voiceprint/data/speech/speaker_diarization/merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130"
CKPT="/root/code/github_repos/LI-MSV-TidyVoice2026-fork/recipes/DeepASV/results/checkpoints/Lora_Adapter_MFA/ckpt_0027_6000item.pth"
YAML="/root/code/github_repos/LI-MSV-TidyVoice2026-fork/recipes/DeepASV/results/checkpoints/Lora_Adapter_MFA/train.yaml"
OUT="${BASE}/embeddings_w2vbert_lora_alldatasets/embeddings_utterances"

for gpu in 0 1 2 3; do
    for proc in 0 1 2 3 4 5 6 7; do
        offset=$((gpu + proc * 4))
        CUDA_VISIBLE_DEVICES=$gpu $PY local/extract_embeddings.py \
            --data_root "${BASE}/audio" \
            --checkpoint "$CKPT" \
            --train_yaml "$YAML" \
            --output_dir "$OUT" \
            --device cuda:0 \
            --file_offset "$offset" \
            --file_stride 32 \
            --num_workers 8 \
            --random_shuffle \
            --skip_existing \
            > /tmp/emb_w2v_gpu${gpu}_p${proc}.log 2>&1 &
    done
done
echo "Launched 32 processes"
wait
echo "=== ALL DONE ==="
