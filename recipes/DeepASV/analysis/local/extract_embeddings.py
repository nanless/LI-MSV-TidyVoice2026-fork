import os, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEEPASV_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '../..'))
_REPO_ROOT = os.path.abspath(os.path.join(_DEEPASV_DIR, '../..'))

sys.path.insert(0, _DEEPASV_DIR)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'deeplab/pretrained/audio2vector/module/transformers/src'))

import argparse
import pickle
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from deeplab.utils.fileio import read_hyperyaml, load_audio


def find_audio_files(data_root, exts=None):
    if exts is None:
        exts = ('.wav', '.flac', '.mp3')
    files = []
    for dirpath, _, filenames in os.walk(data_root):
        for fn in filenames:
            if fn.lower().endswith(exts):
                files.append(os.path.join(dirpath, fn))
    return sorted(files)


def parse_output_subpath(file_path, data_root):
    rel = os.path.relpath(file_path, data_root)
    rel = rel.replace('\\', '/')
    dir_part = os.path.dirname(rel)
    file_stem = os.path.splitext(os.path.basename(rel))[0]
    return dir_part, file_stem


class AudioInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, data_root, sr, max_len):
        self.file_paths = file_paths
        self.data_root = data_root
        self.sr = sr
        self.max_len = max_len

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        signal = load_audio(audio_path, self.sr)[0][:self.max_len]
        signal = torch.from_numpy(signal.astype(np.float32))
        dir_part, file_stem = parse_output_subpath(audio_path, self.data_root)
        return audio_path, dir_part, file_stem, signal


def load_model_and_config(ckpt_path, yaml_path, device):
    ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hparams = read_hyperyaml(path=yaml_path)
    modules = hparams['modules']

    for key, module in modules.items():
        if key == 'classifier':
            continue
        if key not in ckpt_data['modules']:
            print(f'      {key}: <Not found in checkpoint, keeping init weights>')
            module = module.eval().to(device)
            continue
        curr_state_dict = module.state_dict()
        ckpt_state_dict = ckpt_data['modules'][key]
        mismatched = False
        for k in curr_state_dict.keys():
            if k in ckpt_state_dict and curr_state_dict[k].shape == ckpt_state_dict[k].shape:
                curr_state_dict[k] = ckpt_state_dict[k]
            else:
                mismatched = True
        module.load_state_dict(curr_state_dict)
        module = module.eval().to(device)
        if mismatched:
            print(f'      {key}: <Partial weights matched>')
        else:
            print(f'      {key}: <All weights matched>')
    return modules, hparams


def extract_embd(model, inputs):
    out = model(inputs)
    if isinstance(out, (tuple, list)):
        embd = out[0]
    else:
        embd = out
    return embd


def main():
    parser = argparse.ArgumentParser(description='提取 W2V-BERT 说话人嵌入')
    parser.add_argument('--data_root', required=True, help='音频数据根目录')
    parser.add_argument('--checkpoint', required=True, help='模型断点路径 (.pth)')
    parser.add_argument('--train_yaml', required=True, help='训练配置文件路径 (train.yaml)')
    parser.add_argument('--output_dir', required=True, help='输出嵌入的目录')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--skip_existing', action='store_true', help='跳过已有嵌入的音频')
    parser.add_argument('--random_shuffle', action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--max_files', type=int, default=0, help='最大文件数，0 表示全部')
    parser.add_argument('--file_stride', type=int, default=1, help='多进程分片步长')
    parser.add_argument('--file_offset', type=int, default=0, help='多进程分片偏移')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'从以下路径加载模型: {args.checkpoint}')
    print(f'从以下路径加载配置: {args.train_yaml}')
    modules, hparams = load_model_and_config(args.checkpoint, args.train_yaml, args.device)

    sr = hparams['sample_rate']
    max_len = int(hparams['max_valid_dur'] * sr) if 'max_valid_dur' in hparams else sr * 60
    dtype = torch.bfloat16 if hparams.get('use_amp', True) else torch.float32

    spk_model = modules['spk_model']

    audio_files = find_audio_files(args.data_root)
    print(f'找到 {len(audio_files)} 个音频文件')

    if args.random_shuffle:
        rng = np.random.RandomState(args.random_seed)
        rng.shuffle(audio_files)

    if args.max_files > 0:
        audio_files = audio_files[:args.max_files]

    audio_files = audio_files[args.file_offset::args.file_stride]
    print(f'GPU 分片 [{args.file_offset}/{args.file_stride}]: {len(audio_files)} 个文件 '
          f'(num_workers={args.num_workers})')

    dataset = AudioInferenceDataset(audio_files, args.data_root, sr, max_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        pin_memory_device=str(args.device),
        prefetch_factor=8 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    processed = 0
    skipped = 0
    write_executor = ThreadPoolExecutor(max_workers=2)
    pending = []

    def _write_one(out_path, emb):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'wb') as f:
            pickle.dump(emb, f)

    for audio_path, dir_part, file_stem, aud_inputs in tqdm(loader, desc=f'GPU{args.file_offset}', ncols=100):
        audio_path = audio_path[0]
        dir_part = dir_part[0]
        file_stem = file_stem[0]
        aud_inputs = aud_inputs.to(args.device, non_blocking=True)

        out_path = os.path.join(args.output_dir, dir_part, f'{file_stem}.pkl')
        if args.skip_existing and os.path.exists(out_path):
            skipped += 1
            continue

        with torch.autocast('cuda', dtype=dtype):
            with torch.no_grad():
                embedding = extract_embd(spk_model, aud_inputs)
                if len(embedding.shape) == 3:
                    embedding = embedding[:, -1, :]
                embedding = embedding.float().detach().cpu().numpy().squeeze(0)

        pending.append(write_executor.submit(_write_one, out_path, embedding))
        processed += 1

        if len(pending) > 200:
            pending = [f for f in pending if not f.done()]

    for f in pending:
        f.result()

    write_executor.shutdown(wait=True)
    print(f'\n完成。已处理: {processed}, 已跳过: {skipped}, 总文件数: {len(audio_files)}')
    print(f'嵌入已保存到: {args.output_dir}')


if __name__ == '__main__':
    main()
