import os
import sys
import argparse
import pickle
import glob
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def find_leaf_dirs_with_pkl(root_dir):
    dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(fn.endswith('.pkl') for fn in filenames):
            dirs.append(dirpath)
    return sorted(dirs)


def count_utterances(speaker_dir, exclude_prefix=None, exclude_pattern=None):
    files = glob.glob(os.path.join(speaker_dir, '*.pkl'))
    basenames = [os.path.basename(f) for f in files]

    if exclude_prefix:
        basenames = [b for b in basenames if not b.startswith(exclude_prefix)]
    if exclude_pattern:
        import fnmatch
        basenames = [b for b in basenames if not fnmatch.fnmatch(b, exclude_pattern)]

    return len(basenames)


def compute_speaker_embedding(args_pack):
    speaker_dir, utterances_dir, speakers_dir, min_utterances, skip_existing, exclude_prefix, exclude_pattern = args_pack

    rel_dir = os.path.relpath(speaker_dir, utterances_dir)
    speaker_id = rel_dir
    out_path = os.path.join(speakers_dir, rel_dir, f'{os.path.basename(speaker_dir)}.pkl')

    if skip_existing and os.path.exists(out_path):
        return {'speaker_id': speaker_id, 'status': 'skipped', 'num_utterances': 0}

    import fnmatch
    embedding_files = sorted(glob.glob(os.path.join(speaker_dir, '*.pkl')))
    basenames = [os.path.basename(f) for f in embedding_files]

    if exclude_prefix:
        embedding_files = [f for f, b in zip(embedding_files, basenames) if not b.startswith(exclude_prefix)]
        basenames = [b for b in basenames if not b.startswith(exclude_prefix)]
    if exclude_pattern:
        embedding_files = [f for f, b in zip(embedding_files, basenames) if not fnmatch.fnmatch(b, exclude_pattern)]

    num_utterances = len(embedding_files)
    if num_utterances < min_utterances:
        return {'speaker_id': speaker_id, 'status': 'too_few', 'num_utterances': num_utterances}

    embeddings = []
    for emb_file in embedding_files:
        try:
            with open(emb_file, 'rb') as f:
                emb = pickle.load(f)
            embeddings.append(emb)
        except Exception as e:
            tqdm.write(f'加载失败 {emb_file}: {e}')

    if len(embeddings) == 0:
        return {'speaker_id': speaker_id, 'status': 'load_error', 'num_utterances': 0}

    try:
        avg_embedding = np.mean(np.stack(embeddings, axis=0), axis=0)
    except (ValueError, TypeError) as e:
        tqdm.write(f'形状不匹配 {speaker_id}: {e}')
        return {'speaker_id': speaker_id, 'status': 'shape_mismatch', 'num_utterances': num_utterances}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(avg_embedding.astype(np.float32), f)

    return {'speaker_id': speaker_id, 'status': 'ok', 'num_utterances': num_utterances}


def main():
    parser = argparse.ArgumentParser(description='通过平均所有语句嵌入计算说话人级别的嵌入')
    parser.add_argument('--utterances_dir', required=True, help='语句嵌入目录（step1 输出）')
    parser.add_argument('--speakers_dir', required=True, help='说话人嵌入输出目录')
    parser.add_argument('--min_utterances', type=int, default=1, help='每个说话人最少需要的语句数')
    parser.add_argument('--num_processes', type=int, default=0, help='进程数（0 = 自动）')
    parser.add_argument('--chunk_size', type=int, default=10, help='并行处理的 chunk 大小')
    parser.add_argument('--skip_existing', action='store_true', help='跳过已有嵌入的说话人')
    parser.add_argument('--exclude_filename_prefix', default=None, help='排除文件名以此前缀开头的文件')
    parser.add_argument('--exclude_filename_pattern', default=None, help='排除匹配此 glob 模式的文件')
    args = parser.parse_args()

    if args.num_processes <= 0:
        args.num_processes = cpu_count()
    print(f'使用 {args.num_processes} 个进程')

    os.makedirs(args.speakers_dir, exist_ok=True)

    leaf_dirs = find_leaf_dirs_with_pkl(args.utterances_dir)
    print(f'找到 {len(leaf_dirs)} 个说话人目录')

    total_utterances = sum(
        count_utterances(d, args.exclude_filename_prefix, args.exclude_filename_pattern)
        for d in leaf_dirs
    )
    print(f'总共 {total_utterances} 个语句文件')

    task_args = [
        (sd, args.utterances_dir, args.speakers_dir, args.min_utterances, args.skip_existing,
         args.exclude_filename_prefix, args.exclude_filename_pattern)
        for sd in leaf_dirs
    ]

    results = []
    with Pool(processes=args.num_processes) as pool:
        for result in tqdm(pool.imap_unordered(compute_speaker_embedding, task_args, chunksize=args.chunk_size),
                           total=len(task_args), desc='计算说话人嵌入', ncols=100):
            results.append(result)

    ok = [r for r in results if r['status'] == 'ok']
    skipped = [r for r in results if r['status'] == 'skipped']
    too_few = [r for r in results if r['status'] == 'too_few']
    errors = [r for r in results if r['status'] == 'load_error']
    shape_err = [r for r in results if r['status'] == 'shape_mismatch']

    print(f'\n完成。成功: {len(ok)}, 跳过: {len(skipped)}, 语句不足: {len(too_few)}, 加载错误: {len(errors)}, 形状不匹配: {len(shape_err)}')

    if ok:
        total_utts_used = sum(r['num_utterances'] for r in ok)
        print(f'每个说话人平均语句数: {total_utts_used / len(ok):.1f}')


if __name__ == '__main__':
    main()
