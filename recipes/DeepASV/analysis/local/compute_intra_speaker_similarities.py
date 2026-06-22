import os
import argparse
import pickle
import glob
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def find_leaf_dirs_with_pkl(root_dir):
    dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(fn.endswith('.pkl') for fn in filenames):
            dirs.append(dirpath)
    return sorted(dirs)


def load_utterance_embeddings(speaker_dir):
    emb_files = sorted(glob.glob(os.path.join(speaker_dir, '*.pkl')))
    if len(emb_files) == 0:
        return None, [], None

    embeddings = []
    for emb_file in emb_files:
        try:
            with open(emb_file, 'rb') as f:
                emb = pickle.load(f)
            if isinstance(emb, dict):
                emb = emb['embedding']
            embeddings.append(emb.squeeze().astype(np.float64))
        except Exception:
            pass

    if len(embeddings) < 2:
        return None, [os.path.basename(f) for f in emb_files[:1]], None

    emb_matrix = np.stack(embeddings, axis=0)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms

    return emb_matrix, [os.path.basename(f) for f in emb_files], None


def compute_pairwise_similarities(emb_matrix):
    n = emb_matrix.shape[0]
    sims = emb_matrix @ emb_matrix.T
    i_upper = np.triu_indices(n, k=1)
    sims_flat = sims[i_upper].astype(np.float64)
    return sims_flat


def bin_distribution(sims, num_bins=200):
    if len(sims) == 0:
        return [], [], []
    hist, bin_edges = np.histogram(sims, bins=num_bins, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return hist.astype(int).tolist(), bin_edges.tolist(), bin_centers.tolist()


def compute_speaker_intra_stats(args_pack):
    speaker_dir, utterances_dir, output_dir, max_utterances, skip_existing = args_pack

    rel_dir = os.path.relpath(speaker_dir, utterances_dir)
    speaker_id = rel_dir if rel_dir != '.' else os.path.basename(speaker_dir)
    out_dir = os.path.join(output_dir, rel_dir)
    stats_path = os.path.join(out_dir, f'intra_speaker_stats.json')
    hist_path = os.path.join(out_dir, f'intra_speaker_hist.pkl')

    if skip_existing and os.path.exists(stats_path) and os.path.exists(hist_path):
        return {'speaker_id': speaker_id, 'status': 'skipped'}

    emb_matrix, file_basenames, err = load_utterance_embeddings(speaker_dir)
    if emb_matrix is None:
        loaded = len(file_basenames)
        if loaded < 2:
            return {'speaker_id': speaker_id, 'status': 'too_few_utterances', 'num_utterances': loaded}
        else:
            return {'speaker_id': speaker_id, 'status': 'load_error', 'num_utterances': 0}

    n_utt = emb_matrix.shape[0]

    if n_utt > max_utterances:
        rng = np.random.RandomState(hash(speaker_id) % (2**31))
        sampled_idx = rng.choice(n_utt, size=max_utterances, replace=False)
        emb_matrix = emb_matrix[sampled_idx]
        n_utt_sampled = max_utterances
        sampled = True
    else:
        n_utt_sampled = n_utt
        sampled = False

    sims = compute_pairwise_similarities(emb_matrix)
    n_pairs = len(sims)

    hist, bin_edges, bin_centers = bin_distribution(sims)

    stats = {
        'speaker_id': speaker_id,
        'num_utterances': int(n_utt),
        'num_utterances_used': int(n_utt_sampled),
        'sampled': sampled,
        'num_pairs': int(n_pairs),
        'mean_similarity': round(float(np.mean(sims)), 6),
        'std_similarity': round(float(np.std(sims)), 6),
        'min_similarity': round(float(np.min(sims)), 6),
        'max_similarity': round(float(np.max(sims)), 6),
        'median_similarity': round(float(np.median(sims)), 6),
        'q25': round(float(np.percentile(sims, 25)), 6),
        'q75': round(float(np.percentile(sims, 75)), 6),
        'q05': round(float(np.percentile(sims, 5)), 6),
        'q95': round(float(np.percentile(sims, 95)), 6),
        'histogram': hist,
        'bin_edges': [round(float(e), 4) for e in bin_edges],
        'bin_centers': [round(float(c), 4) for c in bin_centers],
    }

    os.makedirs(out_dir, exist_ok=True)

    hist_data = {
        'hist': hist,
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'sims_flat': sims.astype(np.float32),
    }
    with open(hist_path, 'wb') as f:
        pickle.dump(hist_data, f)

    with open(stats_path, 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    return {'speaker_id': speaker_id, 'status': 'ok', **stats}


def main():
    parser = argparse.ArgumentParser(description='计算每个说话人自身音频间的两两相似度分布')
    parser.add_argument('--utterances_dir', required=True, help='语句嵌入目录（extract_embeddings.py 的输出）')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--max_utterances', type=int, default=1000, help='每个说话人最多使用的语句数（超出则随机采样）')
    parser.add_argument('--num_processes', type=int, default=0, help='进程数（0 = 自动）')
    parser.add_argument('--chunk_size', type=int, default=10, help='并行处理的 chunk 大小')
    parser.add_argument('--skip_existing', action='store_true', help='跳过已有输出文件的说话人')
    parser.add_argument('--max_speakers', type=int, default=0, help='最大说话人数（0 = 全部，用于快速测试）')
    args = parser.parse_args()

    if args.num_processes <= 0:
        args.num_processes = cpu_count()
    print(f'使用 {args.num_processes} 个进程')

    os.makedirs(args.output_dir, exist_ok=True)

    leaf_dirs = find_leaf_dirs_with_pkl(args.utterances_dir)
    print(f'找到 {len(leaf_dirs)} 个说话人目录')

    if args.max_speakers > 0 and args.max_speakers < len(leaf_dirs):
        rng = np.random.RandomState(42)
        leaf_dirs = sorted(rng.choice(leaf_dirs, size=args.max_speakers, replace=False).tolist())
        print(f'限制为 {len(leaf_dirs)} 个说话人（随机采样）')

    print(f'最大语句数/说话人: {args.max_utterances}')

    task_args = [
        (sd, args.utterances_dir, args.output_dir, args.max_utterances, args.skip_existing)
        for sd in leaf_dirs
    ]

    results = []
    with Pool(processes=args.num_processes) as pool:
        for result in tqdm(pool.imap_unordered(compute_speaker_intra_stats, task_args, chunksize=args.chunk_size),
                           total=len(task_args), desc='计算说话人内相似度', ncols=100):
            results.append(result)

    ok = [r for r in results if r['status'] == 'ok']
    skipped = [r for r in results if r['status'] == 'skipped']
    too_few = [r for r in results if r['status'] == 'too_few_utterances']
    errors = [r for r in results if r['status'] == 'load_error']

    print(f'\n完成。成功: {len(ok)}, 跳过: {len(skipped)}, 语句太少（<2）: {len(too_few)}, 加载错误: {len(errors)}')

    if not ok:
        print('没有成功处理的说话人。')
        return

    all_means = np.array([r['mean_similarity'] for r in ok], dtype=np.float64)
    all_stds = np.array([r['std_similarity'] for r in ok], dtype=np.float64)
    all_medians = np.array([r['median_similarity'] for r in ok], dtype=np.float64)
    all_mins = np.array([r['min_similarity'] for r in ok], dtype=np.float64)
    all_maxs = np.array([r['max_similarity'] for r in ok], dtype=np.float64)

    sampled_count = sum(1 for r in ok if r['sampled'])
    total_utts_list = np.array([r['num_utterances'] for r in ok])

    aggregated_hist = np.zeros(200, dtype=np.float64)
    for r in ok:
        aggregated_hist += np.array(r['histogram'], dtype=np.float64)
    aggregated_hist = aggregated_hist.tolist()

    summary = {
        'total_speakers_processed': len(ok),
        'total_speakers_found': len(leaf_dirs),
        'speakers_skipped': len(skipped),
        'speakers_too_few_utterances': len(too_few),
        'speakers_load_error': len(errors),
        'speakers_sampled': sampled_count,
        'max_utterances_per_speaker': args.max_utterances,
        'num_utterances_per_speaker': {
            'min': int(np.min(total_utts_list)),
            'max': int(np.max(total_utts_list)),
            'mean': round(float(np.mean(total_utts_list)), 1),
            'median': int(np.median(total_utts_list)),
        },
        'mean_similarity_across_speakers': {
            'mean': round(float(np.mean(all_means)), 6),
            'std': round(float(np.std(all_means)), 6),
            'min': round(float(np.min(all_means)), 6),
            'max': round(float(np.max(all_means)), 6),
            'median': round(float(np.median(all_means)), 6),
        },
        'std_similarity_across_speakers': {
            'mean': round(float(np.mean(all_stds)), 6),
            'std': round(float(np.std(all_stds)), 6),
            'min': round(float(np.min(all_stds)), 6),
            'max': round(float(np.max(all_stds)), 6),
            'median': round(float(np.median(all_stds)), 6),
        },
        'median_similarity_across_speakers': {
            'mean': round(float(np.mean(all_medians)), 6),
            'std': round(float(np.std(all_medians)), 6),
        },
        'aggregated_histogram': aggregated_hist,
        'per_speaker_stats': ok,
    }

    summary_path = os.path.join(args.output_dir, 'intra_speaker_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'\n摘要已保存到: {summary_path}')

    print(f'\n===== 说话人内相似度摘要 =====')
    print(f'处理说话人数: {len(ok)}')
    print(f'被采样的说话人（语句 > {args.max_utterances}）: {sampled_count}')
    print(f'语句数/说话人: 最少 {summary["num_utterances_per_speaker"]["min"]}, '
          f'最多 {summary["num_utterances_per_speaker"]["max"]}, '
          f'平均 {summary["num_utterances_per_speaker"]["mean"]:.1f}, '
          f'中位数 {summary["num_utterances_per_speaker"]["median"]}')
    print(f'各说话人平均相似度的分布: '
          f'均值 {summary["mean_similarity_across_speakers"]["mean"]:.4f}, '
          f'标准差 {summary["mean_similarity_across_speakers"]["std"]:.4f}, '
          f'范围 [{summary["mean_similarity_across_speakers"]["min"]:.4f}, {summary["mean_similarity_across_speakers"]["max"]:.4f}]')


if __name__ == '__main__':
    main()
