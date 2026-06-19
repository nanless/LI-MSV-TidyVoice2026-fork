# AGENTS.md — LI-MSV-TidyVoice2026

语言无关的多语言说话人验证（W2V-BERT 2.0 骨干 + LoRA + MFA + GRL）。

## 快速命令

```bash
# 环境配置（需要 conda + conda-forge 安装 sox）
conda create -y -n asv python=3.9
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip uninstall transformers    # 必须卸载 pip 版本；使用项目内置版本
conda install -c conda-forge sox

# Stage1 第一步：LoRA 冻结预训练（ArcFace）
cd recipes/DeepASV
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag base_s1_ --is_distributed true --yaml conf/base_model/s1.yaml

# Stage1 第二步：将 LoRA 合并到基础模型
cd recipes/DeepASV/utils && python3 lora_merge.py

# Stage1 第三步：联合微调（解冻）
cd recipes/DeepASV
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag base_s2_ --is_distributed true --yaml conf/base_model/s2.yaml \
  --pretrain results/checkpoints/base_s1_XXX/merge_lora.pth

# Stage2 第零步：用平均嵌入初始化分类器（仅当单独在 TidyVoice 上微调时需要）
cd recipes/DeepASV/utils && python3 init_classifier.py

# Stage2 第一步：训练语言分类器
cd recipes/DeepASV
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12886 train_sf2_lang.py \
  --tag base_ft_language_ --is_distributed true --yaml conf/ft_base_model/s1.yaml \
  --pretrain results/checkpoints/base_s2_XXX/ckpt_only_tv.pth

# Stage2 第二步：GRL + SphereFace2 联合训练
cd recipes/DeepASV
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12886 train_sf2_lang_grl.py \
  --tag base_ft_language_grl_ --is_distributed true --yaml conf/ft_base_model/s2.yaml \
  --pretrain results/checkpoints/base_ft_language_XXX/merged.pth

# 嵌入提取 + EER 评估
cd recipes/DeepASV/utils && python3 get_embd_w2v.py
```

## 必须了解的架构

```
recipes/DeepASV/       # 训练入口 — 运行前必须先 cd 到此目录
  train.py             # Stage1：ArcFace 分类器训练
  train_sf2.py         # Stage1：SphereFace2 分类器训练
  train_sf2_lang.py    # Stage2-第一步：训练语言分类器（冻结说话人模型）
  train_sf2_lang_grl.py # Stage2-第二步：GRL + SphereFace2 联合训练
  local/
    spk_model.py       # 模型定义（5 种变体）
    spk_classifier.py  # ArcFace、SphereFace2
    dataset.py / dataset_lang.py  # 数据集 + WavBatchSampler
deeplab/               # 核心库
  core/trainer.py      # 基础 Trainer（DDP、AMP、梯度累积、wandb、断点保存）
  pretrained/audio2vector/
    api.py             # AudioEncoder、LoRA 配置工厂函数
    module/transformers/  # 内置 HuggingFace Transformers v4.57.0.dev0
  dataio/              # 音频 I/O、数据增强（噪声、混响、编解码器）
  metric/eer.py        # EER、DET 曲线、minDCF
  utils/               # 文件 I/O（SCP/trial）、语料库加载
```

**模型流水线：** 原始音频 (16kHz) → SeamlessM4TFeatureExtractor (80 维 mel FBANK) → W2V-BERT 2.0（24 层 Conformer，1024 维）→ 每层 Adapter（1024→128→128）→ MFA 拼接（128×24=3072 维）→ ASP 池化 → bottleneck（256 维说话人嵌入）。

- 配置中 `n_mfa_layers = -1` 时，使用全部 24 层 Conformer。
- `Audio2Vec_based_Adapter` — Stage1 使用；添加 adapter + ASP 池化。
- `Audio2Vec_based_Adapter_Language` — Stage2 使用；添加 GRL + 语言分类头。
- `GrlLayer` 反转梯度：`grad ← -λ × grad`（λ = 0.1）。

## 关键路径/陷阱细节

**所有脚本必须在 `recipes/DeepASV/` 目录下运行。** 每个训练脚本开头都有 `sys.path.append('../..')` 和 `sys.path.append('../../deeplab/pretrained/audio2vector/module/transformers/src')`。内置 transformers 必须在 Python 路径中。

**必须使用内置 transformers，不能使用 pip 版本。** 执行 `pip install -r requirements.txt` 后必须 `pip uninstall transformers`。若两者同时存在会导致导入冲突。

**配置文件使用 `hyperpyyaml`**，支持自定义标签：`!apply:fn [args]`、`!name:ClassName`、`!new:ClassName`、`!ref <key>`。配置文件是可执行的——它们直接构造 Python 对象。

**所有 YAML 配置和工具脚本中的绝对路径硬编码** 均指向 `/work/zl389/...`。需要更新 `corpus_dir`、`musan_path`、`rirs_path`、`train_data` 路径、`valid_data` 的 SCP/trial 路径、`train_utt2lang`/`test_utt2lang`，以及 `utils/*.py` 中的所有 ckpt 路径。

**断点格式：** `dict(modules={'spk_model': state_dict, 'classifier': state_dict}, epoch_idx=N)`。Trainer 中的 `load_checkpoints` 方法会优雅处理分类器权重的形状不匹配（截断或填充）——这对使用不同数量说话人进行微调至关重要。

**数据格式：**
- SCP 文件：制表符分隔 `utt_id\twav_path`（2 个字段，解析为 `{reco, wav_path}`）
- Trial 文件：空格分隔 `key utt1 utt2`
- `utt2lang` 文件：空格分隔 `utt_id language_label`
- 训练语料库通过 `load_audio_corpus(root_dir, [subdirs])` 加载 → `{spk_id: [wav_paths]}`
- `WavBatchSampler` 向数据集传递 `(idx, dur)` 元组；数据集同时接收两者。

**`classifier.out_features` 必须等于训练说话人数量**（考虑速度扰动扩展后）。不匹配会在 `prep()` 中被 assert 断言。对于仅 TidyVoice 的微调，用 `init_classifier.py` 以平均嵌入初始化分类器。

**SphereFace2 返回 `(output, loss)` 元组**，而 ArcFace 只返回 `output`。各训练脚本对此处理不同——`train.py`（ArcFace）对输出调用 `F.cross_entropy()`，而 `train_sf2.py`（SphereFace2）直接使用内部 loss。

**混合精度使用 bfloat16**（不是 float16）。由配置中的 `use_amp: true` 控制。

**必须安装 SoX**，用于 `load_audio()` 中的音频重采样（通过 `torchaudio.sox_effects`）。用 `conda install -c conda-forge sox` 安装。

**无测试、无 CI、无 linter 配置。** 验证通过在 VoxCeleb/TidyVoice trial 文件上计算 EER 完成，使用 `get_embd_w2v.py`。

**训练需要 CUDA GPU**——不支持 CPU。分布式训练使用 NCCL 后端，通过 `torchrun` 启动。单 GPU 模式：省略 `--is_distributed`（默认 False），直接用 `python3 train.py ...` 运行，不需要 `torchrun`。

**`items_save: true`** 配置项在每个 `item_save_steps` 次迭代时启用训练中验证和断点保存（大规模训练时开销较大）。

## 合成数据生成（可选）

```
python3 m2m100.py   # M2M100 翻译（英语 → 9 种目标语言）
python3 whisper.py  # Whisper-large-v3 ASR 获取参考音频
python3 tts.py      # Qwen3-TTS 语音克隆（10 种语言）
```

这些脚本在 `recipes/DeepASV/Syn/` 目录下，需要各自的模型依赖（transformers 等）。

## 参考文献

论文：[Language-Invariant Multilingual Speaker Verification for the TidyVoice 2026 Challenge](https://arxiv.org/abs/2603.08092)
基于：[ZXHY-82/w2v-BERT-2.0_SV](https://github.com/ZXHY-82/w2v-BERT-2.0_SV)
许可证：CC BY-NC-SA 4.0
