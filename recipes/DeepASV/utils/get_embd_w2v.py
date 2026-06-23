import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SCRIPT_DIR = Path(__file__).resolve().parent
DEEPASV_DIR = SCRIPT_DIR.parent
REPO_ROOT = DEEPASV_DIR.parent.parent
TRANSFORMERS_SRC = REPO_ROOT / "deeplab/pretrained/audio2vector/module/transformers/src"

sys.path.insert(0, DEEPASV_DIR.as_posix())
sys.path.insert(0, REPO_ROOT.as_posix())
sys.path.insert(0, TRANSFORMERS_SRC.as_posix())

import numpy as np
import torch
from tqdm import tqdm

from deeplab.utils.fileio import read_hyperyaml, load_audio

import warnings

warnings.filterwarnings("ignore")

DEFAULT_BASE = Path(
    "/root/group-shared/voiceprint/data/speech/speaker_diarization/"
    "merged_datasets_20250610_vad_segments_mtfaa_enhanced_extend_kid_withclone_addlibrilight_1130"
)
DEFAULT_CKPT = (
    REPO_ROOT
    / "recipes/DeepASV/results/checkpoints/Lora_Adapter_MFA/ckpt_0027_6000item.pth"
)
DEFAULT_TRAIN_YAML = DEFAULT_CKPT.parent / "train.yaml"
DEFAULT_SE_TRAIN_ROOT = Path("/root/code/gitlab_repos/se_train")


def extract_embd(model, inputs):
    out = model(inputs)
    if isinstance(out, (tuple, list)):
        embd = out[0]
    else:
        embd = out
    return embd


def parse_args():
    parser = argparse.ArgumentParser(description="Extract W2V-BERT speaker embeddings.")
    parser.add_argument("--scp_path", default=None, help="Optional wav.scp with utt<TAB>path rows.")
    parser.add_argument("--data_root", default=(DEFAULT_BASE / "audio").as_posix())
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT.as_posix())
    parser.add_argument("--train_yaml", default=DEFAULT_TRAIN_YAML.as_posix())
    parser.add_argument("--trial_path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_files", type=int, default=40)
    parser.add_argument("--max_scan_files", type=int, default=400)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--compare_se_train", action="store_true")
    parser.add_argument("--se_train_root", default=DEFAULT_SE_TRAIN_ROOT.as_posix())
    parser.add_argument(
        "--base_model_dir",
        default=(REPO_ROOT / "deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0").as_posix(),
    )
    return parser.parse_args()


def collect_scp(args):
    if args.scp_path:
        scp = []
        with open(args.scp_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    scp.append((parts[0], parts[1]))
        return scp[: args.max_files] if args.max_files > 0 else scp

    data_root = Path(args.data_root)
    audio_paths = []
    for dirpath, dirnames, filenames in os.walk(data_root):
        dirnames[:] = sorted(dirnames)
        for filename in sorted(filenames):
            if filename.lower().endswith((".wav", ".flac", ".mp3")):
                audio_paths.append(Path(dirpath) / filename)
                if len(audio_paths) >= args.max_scan_files:
                    break
        if len(audio_paths) >= args.max_scan_files:
            break

    rng = random.Random(args.random_seed)
    rng.shuffle(audio_paths)
    if args.max_files > 0:
        audio_paths = audio_paths[: args.max_files]
    return [(path.stem, path.as_posix()) for path in audio_paths]


def load_model(ckpt_path, train_yaml_path, device):
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = read_hyperyaml(path=train_yaml_path)
    modules = hparams["modules"]

    for key, module in modules.items():
        if key == "classifier":
            continue
        curr_state_dict = module.state_dict()
        ckpt_state_dict = ckpt_data["modules"][key]
        mismatched = False
        for k in curr_state_dict.keys():
            if k in ckpt_state_dict and curr_state_dict[k].shape == ckpt_state_dict[k].shape:
                curr_state_dict[k] = ckpt_state_dict[k]
            else:
                mismatched = True
        module.load_state_dict(curr_state_dict)
        module = module.eval().to(device)

        if mismatched:
            print("      {}: <Partial weights matched>".format(key))
        else:
            print("      {}: <All weights matched>".format(key))

    return modules, hparams


def load_se_train_model(args):
    # The original TidyVoice model has already been constructed. Remove its
    # vendored top-level transformers path before importing se_train so this
    # comparison uses the environment's installed transformers package.
    transformers_src = TRANSFORMERS_SRC.as_posix()
    sys.path[:] = [p for p in sys.path if p != transformers_src]
    for name in list(sys.modules):
        if name == "transformers" or name.startswith("transformers."):
            del sys.modules[name]

    se_train_root = Path(args.se_train_root).resolve().as_posix()
    if se_train_root not in sys.path:
        sys.path.insert(0, se_train_root)

    from speech_enhancement.runners.pse.model.speaker_embedding.w2vbert_lora_sv.features import (
        compute_fbank_w2vbert,
    )
    from speech_enhancement.runners.pse.model.speaker_embedding.w2vbert_lora_sv.wrapper import (
        W2VBertLoraSVEmbedding,
    )

    model = W2VBertLoraSVEmbedding(
        checkpoint_path=args.checkpoint,
        train_yaml_path=args.train_yaml,
        base_model_dir=args.base_model_dir,
        output_compress_enable=False,
        use_amp=True,
    ).to(args.device)
    model.eval()
    return model, compute_fbank_w2vbert


def main():
    args = parse_args()
    scp_list = collect_scp(args)
    if not scp_list:
        raise RuntimeError(f"No audio files found under {args.data_root}")

    modules, hparams = load_model(args.checkpoint, args.train_yaml, args.device)
    sr = hparams["sample_rate"]
    max_len = int(hparams["max_valid_dur"] * hparams["sample_rate"])
    dtype = torch.bfloat16 if hparams.get("use_amp", True) else torch.float32
    autocast_enabled = str(args.device).startswith("cuda")

    print(dtype)
    print(f"audio_count={len(scp_list)}")

    se_model = None
    compute_fbank_w2vbert = None
    cos_vals = []
    max_abs_vals = []
    mean_abs_vals = []
    if args.compare_se_train:
        se_model, compute_fbank_w2vbert = load_se_train_model(args)

    utt2embd = {}
    with torch.autocast("cuda", dtype=dtype, enabled=autocast_enabled):
        with torch.no_grad():
            for scp_data in tqdm(scp_list, ncols=80):
                utt = scp_data[0]
                if utt in utt2embd:
                    print("Warning: duplicated utt key.")
                wav_path = scp_data[1]
                signal = load_audio(wav_path, sr)[0].astype(np.float32)[:max_len]
                aud_inputs = torch.from_numpy(signal).float().to(args.device)
                # utt2embd[utt] = modules['spk_model'](aud_inputs).float().detach().cpu().numpy()
                embedding = extract_embd(modules["spk_model"], aud_inputs)
                if len(embedding.shape) == 3:
                    embedding = embedding[:, -1, :]
                embedding = embedding.float().detach().cpu()
                utt2embd[utt] = embedding.numpy()

                if se_model is not None:
                    feats = torch.from_numpy(compute_fbank_w2vbert(signal, sr)).float().unsqueeze(0).to(args.device)
                    batch = {
                        "target_enrollment_feature": feats,
                        "target_enrollment_feature_lengths": torch.tensor(
                            [feats.shape[1]], device=args.device, dtype=torch.long
                        ),
                        "target_enrollment_feature_mask": torch.ones(
                            (1, feats.shape[1]), device=args.device, dtype=torch.bool
                        ),
                    }
                    se_embedding = se_model(batch)["target_embedding"].float().detach().cpu()
                    diff = (embedding - se_embedding).abs()
                    cos = torch.nn.functional.cosine_similarity(embedding, se_embedding, dim=-1).item()
                    cos_vals.append(cos)
                    max_abs_vals.append(diff.max().item())
                    mean_abs_vals.append(diff.mean().item())

    if args.compare_se_train:
        print(
            "se_train_compare: "
            f"n={len(cos_vals)}, "
            f"cos_min={min(cos_vals):.9f}, "
            f"cos_mean={sum(cos_vals) / len(cos_vals):.9f}, "
            f"max_abs_max={max(max_abs_vals):.9g}, "
            f"mean_abs_max={max(mean_abs_vals):.9g}"
        )

    if args.trial_path:
        from deeplab.metric.eer import get_eer

        eer, threshold, _, _ = get_eer(utt2embd, args.trial_path)
        print("EER: {:.4f}%".format(eer * 100))


if __name__ == "__main__":
    main()