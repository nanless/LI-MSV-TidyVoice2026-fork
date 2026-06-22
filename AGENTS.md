# AGENTS.md — LI-MSV-TidyVoice2026

语言无关的多语言说话人验证（W2V-BERT 2.0 骨干 + LoRA + MFA + GRL）。

## 快速命令

```bash
# ===== 环境配置 =====
conda create -y -n asv python=3.9
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip uninstall transformers    # 必须卸载 pip 版本；使用项目内置版本
conda install -c conda-forge sox

# ===== 下载 W2V-BERT 2.0 预训练权重 =====
# 从 https://huggingface.co/facebook/w2v-bert-2.0/blob/main/model.safetensors
# 放到 deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/

# ===== Stage1 第一步：LoRA 冻结预训练（ArcFace）=====
cd recipes/DeepASV
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag base_s1_ --is_distributed true --yaml conf/base_model/s1.yaml

# ===== Stage1 第二步：将 LoRA 合并到基础模型 =====
cd recipes/DeepASV/utils && python3 lora_merge.py

# ===== Stage1 第三步：联合微调（解冻）=====
cd recipes/DeepASV
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
  --tag base_s2_ --is_distributed true --yaml conf/base_model/s2.yaml \
  --pretrain results/checkpoints/base_s1_XXX/merge_lora.pth

# ===== Stage2 第零步：用平均嵌入初始化分类器（仅当单独在 TidyVoice 上微调时需要）=====
cd recipes/DeepASV/utils && python3 init_classifier.py

# ===== Stage2 第一步：训练语言分类器（冻结说话人分支）=====
cd recipes/DeepASV
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12886 train_sf2_lang.py \
  --tag base_ft_language_ --is_distributed true --yaml conf/ft_base_model/s1.yaml \
  --pretrain results/checkpoints/base_s2_XXX/ckpt_only_tv.pth

# ===== Stage2 第 1.5 步：将说话人分类器权重合并到语言训练输出中 =====
cd recipes/DeepASV/utils && python3 merge_classifier.py

# ===== Stage2 第二步：GRL + SphereFace2 联合训练 =====
cd recipes/DeepASV
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12886 train_sf2_lang_grl.py \
  --tag base_ft_language_grl_ --is_distributed true --yaml conf/ft_base_model/s2.yaml \
  --pretrain results/checkpoints/base_ft_language_XXX/ckpt_update.pth

# ===== 嵌入提取 + EER 评估 =====
cd recipes/DeepASV/utils && python3 get_embd_w2v.py
```

## 必须了解的架构

```
recipes/DeepASV/       # 训练入口 — 运行前必须先 cd 到此目录
  train.py             # Stage1：ArcFace 分类器训练
  train_sf2.py         # Stage1：SphereFace2 分类器训练（备用方案）
  train_sf2_lang.py    # Stage2-第一步：训练语言分类器（冻结说话人模型，不包含 classifier 模块）
  train_sf2_lang_grl.py # Stage2-第二步：GRL + SphereFace2 联合训练
  local/
    spk_model.py       # 模型定义（5 种变体）
    spk_classifier.py  # ArcFace、SphereFace2
    dataset.py / dataset_lang.py  # 数据集 + WavBatchSampler
  analysis/local/      # 独立分析脚本（extract_embeddings, compute_speaker_embeddings, compute_speaker_similarities）
deeplab/               # 核心库
  core/trainer.py      # 基础 Trainer（DDP、AMP(bfloat16)、梯度累积、wandb、断点保存/加载）
  pretrained/audio2vector/
    api.py             # AudioEncoder、LoRA 配置工厂函数
    module/transformers/  # 内置 HuggingFace Transformers v4.57.0.dev0（非 pip，必须手动引入路径）
  dataio/              # 音频 I/O（load_audio via torchaudio.sox_effects）、数据增强（噪声、混响、编解码器）
  metric/eer.py        # EER、DET 曲线、minDCF
  utils/               # 文件 I/O（SCP/trial）、语料库加载（load_audio_corpus）
  core/scheduler.py    # WarmupLR_withStepDecay（每个 epoch step）、WarmupCosineScheduler（每次 iter step）
```

**模型流水线：** 原始音频 (16kHz) → SeamlessM4TFeatureExtractor (80 维 mel FBANK) → W2V-BERT 2.0（24 层 Conformer，1024 维隐藏层）→ 每层 Adapter（1024→128→128）→ MFA 拼接（128×24=3072 维）→ ASP 池化（2x 扩展=6144 维）→ bottleneck（6144→256 维说话人嵌入）。

- 配置中 `n_mfa_layers = -1` 时，使用全部 24 层 Conformer。
- `Audio2Vec_based` — 基础模型；无 adapter，MFA 直接拼接原始隐藏状态（1024×24=24576 维），ASP 池化后 bottleneck。
- `Audio2Vec_based_Adapter` — **Stage1 使用**；添加 per-layer adapter（1024→128→128）+ ASP 池化 + bottleneck(256)。Stage1-第一步使用 LoRA（r=64, alpha=128, 目标模块=["linear_q", "linear_v"]），第二步用 `merge_and_unload()` 合并后不再有 LoRA。
- `Audio2Vec_based_Adapter_Language` — **Stage2 使用**；与 Adapter 相同但额外添加 `GradReverse` 梯度反转层（仅在 `GRL=True` 时生效）+ 语言分类头 `lang_head`（256→128→40）。`spk_embd=True` 时 forward 返回 `(embedding, lang_output)` 元组，否则仅返回 `lang_output`。`frozen_spkmodel=True` 时冻结 adapter_layers/pooling/bottleneck（在 forward 中显式调用 `.eval()` 并包裹 `torch.no_grad()`）。
- `GradReverse`（`torch.autograd.Function`）：`backward` 中 `grad ← -λ × grad`，默认 λ = 0.1。
- `Audio2Vec_based_Weighted_ECAPATDNN` — 替代方案；可学习的软最大层权重聚合 + ECAPA-TDNN 骨干。
- `Audio2Vec_based_Prune` — 教师-学生蒸馏，结构剪枝。

### 5 种模型变体的 forward 返回值对照

| 模型类 | forward 返回值 |
|--------|---------------|
| `Audio2Vec_based` | `(B, 256)` 嵌入 |
| `Audio2Vec_based_Adapter` | `(B, 256)` 嵌入 |
| `Audio2Vec_based_Weighted_ECAPATDNN` | `(B, 256)` 嵌入 |
| `Audio2Vec_based_Prune` | `(teacher_hidden, student_hidden)` 元组 |
| `Audio2Vec_based_Adapter_Language` (spk_embd=True) | `((B, 256) 嵌入, (B, 40) lang输出)` |
| `Audio2Vec_based_Adapter_Language` (spk_embd=False) | `(B, 40)` lang输出 |

## 关键路径/陷阱细节

### Python 路径与工作目录

**所有训练脚本必须在 `recipes/DeepASV/` 目录下运行。** 每个训练脚本开头都有：
```python
sys.path.append('../..')
sys.path.append('../../deeplab/pretrained/audio2vector/module/transformers/src')
```
工具脚本（`utils/*.py`）也需从其各自的目录运行，它们有 `sys.path.append('../')`、`sys.path.append('../../../')` 等相对路径。`spk_model.py` 内部也有自己的路径补丁。内置 transformers 必须在 Python 路径中——`deeplab/pretrained/audio2vector/module/transformers/src/` 路径必须可达。

### 内置 Transformers 冲突

**必须使用内置 transformers，不能使用 pip 版本。** 执行 `pip install -r requirements.txt` 后必须 `pip uninstall transformers`。若两者同时存在会导致导入冲突。

### 所有脚本中的绝对路径硬编码

**所有 YAML 配置和工具脚本中的绝对路径** 均指向 `/work/zl389/...`。需要更新的文件：

**YAML 配置文件：** `conf/base_model/s1.yaml`、`conf/base_model/s2.yaml`、`conf/ft_base_model/s1.yaml`、`conf/ft_base_model/s2.yaml`
- `corpus_dir`、`musan_path`、`rirs_path` 路径
- `train_data` 路径和子目录列表
- `valid_data` 的 `scp_path` / `trial_path`
- `train_utt2lang` / `test_utt2lang`（仅 ft_base_model）

**工具脚本：**
- `utils/lora_merge.py`：第31行 `ckpt_path` 硬编码
- `utils/init_classifier.py`：第10行 `.npy` 路径、第22行 `tv_root`、第34行 ckpt 路径、第45行输出路径
- `utils/merge_classifier.py`：第3行和第9行 `ckpt_path` 硬编码、第15行输出路径
- `utils/get_embd_w2v.py`：第12行 `scp_path`、第15行 `ckpt_path`（当前为空字符串！）、第71行 `trial_path`

> **重要：** `get_embd_w2v.py` 的 `ckpt_path` 默认为空字符串 `''`，必须手动编辑才能运行。

### `--is_distributed` 参数的 argparse 陷阱

```python
parser.add_argument("--is_distributed", default=False, type=bool)
```

`bool("false")` 在 Python 中结果为 `True`（因为是非空字符串），所以 `--is_distributed false` 会错误地启用分布式模式。**正确的单 GPU 用法是省略该参数**（默认 `False`），直接用 `python3 train.py ...`，不需要 `torchrun`。

### 配置文件使用 `hyperpyyaml` 且可执行

YAML 配置通过 `load_hyperpyyaml()` 加载，支持自定义标签：
- `!apply:fn [args]` — 调用函数
- `!name:ClassName` — 返回类引用
- `!new:ClassName` — 实例化类
- `!ref <key>` — 引用之前定义的条目

配置文件直接构造 Python 对象（optimizer、scheduler、model 等），而不仅仅是字典。训练时 YAML 内容会被自动拷贝到断点目录下 `train.yaml`，推理时通过 `read_hyperyaml()` 重新加载。

### `encoder_config: 'config_tea.json'` 必须存在

所有 YAML 配置中 `spk_model` 的 `encoder_config: 'config_tea.json'` 指的是 `deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/config_tea.json`。该文件由 `config.py` 在同目录生成（运行前需要 `config.json` 已存在）。该文件为基础 W2V-BERT 配置添加了 `intermediate_size_group`、`conv_group`、`num_attention_heads_group`、`use_feed_forward`、`use_attention`、`hard_concrete` 等每层参数（共 24 层），且 `prune: False`。如有缺失会导致模型初始化失败。

### 断点格式与加载逻辑

**断点格式：**
```python
{
    'modules': {
        'spk_model': state_dict,
        'classifier': state_dict,  # 可选；train_sf2_lang.py 的断点无此键
    },
    'epoch_idx': N,
}
```

**保存规则：**
- 每个 epoch 结束：仅 rank 0 保存 `ckpt_{epoch_idx:04d}.pth`，所有 rank `dist.barrier()`。
- `items_save=True` 时：每 `item_save_steps` 次迭代保存 `ckpt_{epoch_idx:04d}_{iter_idx}item.pth`。
- `train.yaml` 和 `logs.json` 随断点一起保存。

**加载规则（`trainer.load_checkpoints`）：**
1. `classifier` 键：优雅处理形状不匹配——若 `ckpt_len > curr_len` 则截断，若 `curr_len > ckpt_len` 则用断点权重填充前 N 个并保留其余初始化值。bias 一并处理。这对用不同数量说话人微调至关重要。
2. `spk_model` 键：按名称逐键匹配，仅拷贝形状匹配的权重。部分匹配仅警告不报错。
3. `epoch_idx`：设置 `self.init_epoch_idx`，从断点 epoch+1 继续。
4. `logs.json`：加载至断点 epoch 的历史日志。

### 学习率调度器的两种模式

- **`scheduler`（per-epoch）**：在 `trainer.py:352`，每个 epoch 结束（验证之后）调用 `scheduler.step()`。用于 Stage1 第一步和 Stage2 第一步（`WarmupLR_withStepDecay`）。
- **`scheduler_lmft`（per-iteration）**：在 `trainer.py:314-315`，每次 `optimizer.step()` 之后调用 `scheduler_lmft.step()`。用于 Stage1 第三步和 Stage2 第二步（`WarmupCosineScheduler`：预热→余弦衰减→固定 min_lr）。

> 注意：`scheduler_lmft.step()` 只在 `iter_idx % iters_to_accumulate == 0` 时调用（即梯度累积后），而 `scheduler.step()` 无条件在每个 epoch 结束时调用。

### DDP 封装规则

在 `trainer.initialize_training()` 中：
- **可训练模块**（至少一个参数 `requires_grad=True`）：DDP 封装（分布式）或 `nn.DataParallel`（单 GPU），并转换 SyncBatchNorm。
- **不可训练模块**（全部参数 `requires_grad=False`）：仅 `.to(device)`，不封装。
- `find_unused_parameters` 默认为 `False`。当含有被冻结的子模块时（如 `frozen_spkmodel=True`），可能需要设为 `True`（虽然此项目未显式设置）。

### 混合精度使用 bfloat16

由配置中的 `use_amp: true` 控制。AMP 始终使用 `torch.bfloat16`（不是 float16）。代码 `torch.amp.autocast('cuda', torch.bfloat16)` 包裹在 `trainer.py` 的前向传播中。bfloat16 的范围与 float32 相同，无需 loss scaling，但代码仍然使用 `GradScaler`。

### 数据格式

- **SCP 文件：** 制表符分隔 `utt_id\twav_path`（2 字段，解析为 `{reco, wav_path}`）
- **Trial 文件：** 空格分隔 `key utt1 utt2`
- **`utt2lang` 文件：** 空格分隔 `utt_id language_label`
  - **关键：** `Valid_Dataset` 中用 `wav_path`（绝对路径）作为 key 查找 utt2lang，而非 `utt_id`
- **训练语料库：** 通过 `load_audio_corpus(root_dir, [subdirs])` 加载 → `{spk_id: [wav_paths]}`。多个语料库组用 `{group_id}-{spk_id}` 作为唯一 key。
- **`WavBatchSampler`：** 向数据集传递 `(idx, dur)` 元组，其中 dur 是 `dur_range`（如 [2,3]）内的随机值。内部由 `DistributedSampler`（分布式）或 `RandomSampler`（单 GPU）包装。

### `speed_perturbation: []`（空列表）≠ `None`

配置中 `speed_perturbation: []`（空列表）时，代码检查的是 `if self.speed_perturbation is not None`，空列表不是 None，因此 speed perturbation 逻辑会被触发，但由于列表为空，实际不会创建额外的话语。若要完全禁用，需设为 `None` 或删除该键。

### Stage1 完整流程（5 大语料库预训练）

Stage1 在 5 个语料库上训练：
- CN-Celeb1&2（2793 说话人）
- VoxCeleb2 dev（5994 说话人）
- VoxBlink2（111284 说话人）
- 3D-Speaker test+train（240+10000 说话人）
- KeSpeech（27237 说话人）
- 共计 ~157548 说话人

**`classifier.out_features` 必须等于速度扰动扩展前的训练说话人总数**。`train.py` 的 `prep()` 中有 assert 断言（第57行），`train_sf2_lang.py` 和 `train_sf2_lang_grl.py` 中该 assert 被注释掉。

### Stage2 完整流程与模块结构

**关键：Stage2 的三个步骤中模块配置不同。**

| 步骤 | 脚本 | modules 内容 | classifier | spk_model 配置 |
|------|------|-------------|------------|----------------|
| Step1 | `train_sf2_lang.py` + `ft_base_model/s1.yaml` | **仅 `spk_model`**，无 classifier | 无 | `frozen_encoder=True, frozen_spkmodel=True, GRL=False, spk_embd=True` |
| Step1.5 | `merge_classifier.py` | — | — | — |
| Step2 | `train_sf2_lang_grl.py` + `ft_base_model/s2.yaml` | `spk_model` + `classifier` | SphereFace2(out=3666) | `frozen_encoder=False, frozen_spkmodel=False, GRL=True, spk_embd=True` |

**Step1（`train_sf2_lang.py`）特点：**
- `compute_forward` 返回 `{'emb_output': emb_output, 'lang_output': lang_output}`，无 classifier 前向传播。
- `loss_fn` 对 `lang_output` 做 `F.cross_entropy(lang_output, inputs['lang_labels'])`。
- 验证时计算 EER（嵌入）和语言准确率（lang_acc），两者同时记录。
- 断点保存只包含 `spk_model` 的 state_dict，不含 classifier。

**Step1.5（`merge_classifier.py`）：**
- 从 Stage1 的 `ckpt_only_tv.pth`（ArcFace 分类器权重已用平均嵌入初始化）提取 classifier 权重。
- 将 classifier 权重注入到 Step1 的输出断点中。
- 输出 `ckpt_update.pth`，作为 Step2 的预训练。

**Step2（`train_sf2_lang_grl.py`）特点：**
- `compute_forward` 返回 `{'emb_output, spk_output, lang_output, spk_loss_output, lang_loss}`。
- `loss_fn` 返回 `{'loss_spk': spk_loss_output, 'loss_lang': lang_loss}`，两者求和后 backward。
- `lang_loss = F.cross_entropy(lang_output, lang_labels) * lang_loss_lambda`（lambda=0.1）。
- `spk_loss_output` 来自 SphereFace2 内部（返回 `(output, loss)` 元组）。
- 验证时同样计算 EER 和语言准确率。

### `train_sf2_lang.py` 的 `Valid_Dataset` 语言标签映射细节

`dataset_lang.py` 的 `Valid_Dataset.__init__` 中有两步读取：
1. 先读取 `hparams['train_utt2lang']` 构建 `lang2label` 映射（**语言标签空间由训练数据定义**）
2. 再读取 `hparams['test_utt2lang']` 做实际话语→语言映射

`__getitem__` 中用 `wav_path`（而非 `utt_id`）作为 key 从 `utt2lang` 查找语言标签。

### `get_embd_w2v.py` 使用 `read_hyperyaml` 重载训练配置

嵌入提取脚本不直接读 YAML，而是从断点目录的 `train.yaml`（训练时自动保存的副本）通过 `read_hyperyaml()` 加载 hparams。这要求断点目录下必须存在 `train.yaml` 文件。

### `train_sf2_lang_grl.py` 的 `compute_forward` 中有 3D 嵌入重塑逻辑

```python
if len(emb_output.shape) == 3:
    bsz, seqlen, _ = emb_output.shape
    emb_output = emb_output.reshape(bsz * seqlen, -1)
    inputs['spk_labels'] = inputs['spk_labels'].unsqueeze(1).repeat(1, seqlen).reshape(bsz * seqlen)
```
当模型返回 3D 嵌入（`(B, T, D)`）时，会自动重塑并扩展标签。此逻辑在 `train_sf2_lang.py` 中被注释掉了。

### `valid_batch_size` 默认回退

`trainer.py:122` 中，若 `valid_batch_size` 不在 hparams 中，回退到 `batch_size`。但所有配置都显式设置 `valid_batch_size: 1`（嵌入提取为单话语）。

### SoX 必须安装

`load_audio()` 中通过 `torchaudio.sox_effects` 进行音频重采样。用 `conda install -c conda-forge sox` 安装。缺失时音频加载会失败。

### 训练需要 CUDA GPU

不支持 CPU。分布式训练使用 NCCL 后端通过 `torchrun` 启动。单 GPU 模式：省略 `--is_distributed`，直接 `python3 train.py ...`。

### `items_save: true` 配置项

在每个 `item_save_steps` 次迭代时启用训练中验证和断点保存。`ft_base_model/s2.yaml` 中开启，每 300 步保存。`base_model/s1.yaml` 中关闭。大规模数据集上此功能开销较大。

### `OMP_NUM_THREADS` 与 `num_workers`

`OMP_NUM_THREADS="16"` 设置 CPU OpenMP 线程数，`num_workers: 16` 是 DataLoader 的工作线程数。需要足够的 CPU 核心（建议 ≥16）。

### `torch.load` 需要 `weights_only=False`

项目中多个脚本使用 `torch.load(..., weights_only=False)`。Python 3.12+ 默认 `weights_only=True`，因此这些脚本显式传入 `weights_only=False`。若要在 Python 3.12+ 环境运行，确保所有 `torch.load` 都包含此参数。

### 无测试、无 CI、无 linter 配置

验证通过在 VoxCeleb/TidyVoice trial 文件上计算 EER 完成，使用 `get_embd_w2v.py`。

### YAML 配置中的路径拼写错误

`ft_base_model/s1.yaml` 和 `ft_base_model/s2.yaml` 的 `valid_data` 中：
```
trial_path: '/work/zl389/workspace/LLM_ASV/data/TidyVoice/dev/TidyVocieX_Dev_trialPairs.txt'
```
注意 `TidyVocieX` 是拼写错误（应为 `TidyVoiceX`），这是原始作者数据中的实际文件名。

## Stage2 断点合并流程（详细）

```text
Stage1 base_s2 ckpt (has ArcFace classifier)
      │
      ├── init_classifier.py
      │   - 加载预提取的 tv_embd (.npy)
      │   - 对每个说话人计算平均嵌入
      │   - 用平均嵌入替换 ArcFace 分类器权重
      │   └── 输出: ckpt_only_tv.pth
      │
      ├── train_sf2_lang.py (Step1)
      │   - 仅训练语言分类头
      │   - 模块：仅 spk_model（无 classifier）
      │   └── 输出: ckpt_NNNN.pth（不含 classifier 权重）
      │
      ├── merge_classifier.py (Step1.5)
      │   - 从 ckpt_only_tv.pth 提取 classifier 权重
      │   - 注入到 ckpt_NNNN.pth
      │   └── 输出: ckpt_update.pth
      │
      └── train_sf2_lang_grl.py (Step2)
          - 用 ckpt_update.pth 初始化
          - GRL + SphereFace2 联合训练
```

## 合成数据生成（可选）

```bash
cd recipes/DeepASV/Syn
python3 m2m100.py   # M2M100 翻译（英语 → 9 种目标语言）
python3 whisper.py  # Whisper-large-v3 ASR 获取参考音频
python3 tts.py      # Qwen3-TTS 语音克隆（10 种语言）
```

## 参考文献

论文：[Language-Invariant Multilingual Speaker Verification for the TidyVoice 2026 Challenge](https://arxiv.org/abs/2603.08092)
基于：[ZXHY-82/w2v-BERT-2.0_SV](https://github.com/ZXHY-82/w2v-BERT-2.0_SV)
许可证：CC BY-NC-SA 4.0
