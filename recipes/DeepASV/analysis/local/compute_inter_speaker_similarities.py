import os
import argparse
import pickle
import glob
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def load_speaker_embeddings(speakers_dir):
    emb_files = sorted(glob.glob(os.path.join(speakers_dir, '**', '*.pkl'), recursive=True))
    if len(emb_files) == 0:
        return np.array([]), [], []

    speaker_ids = []
    embeddings = []
    for emb_file in emb_files:
        try:
            with open(emb_file, 'rb') as f:
                emb = pickle.load(f)
            if isinstance(emb, dict):
                emb = emb['embedding']
            embeddings.append(emb.squeeze().astype(np.float64))
            speaker_ids.append(os.path.relpath(os.path.dirname(emb_file), speakers_dir))
        except Exception:
            pass

    if len(embeddings) < 2:
        return np.array([]), [], []

    emb_matrix = np.stack(embeddings, axis=0)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms

    return emb_matrix, speaker_ids, [os.path.basename(f) for f in emb_files]


def compute_chunk_top_sims(args_pack):
    chunk_indices, emb_matrix, top_k = args_pack
    results = []
    for i in chunk_indices:
        row = emb_matrix[i:i+1]
        sims = (row @ emb_matrix.T).squeeze(0)
        sorted_idx = np.argsort(sims)[::-1]
        top = []
        for j in sorted_idx:
            if i == j:
                continue
            top.append({'idx': int(j), 'similarity': float(sims[j])})
            if len(top) >= top_k:
                break
        results.append({'query_idx': int(i), 'top_k': top})
    return results


def main():
    parser = argparse.ArgumentParser(description='评估说话人间的余弦相似度分布')
    parser.add_argument('--embeddings_dir', required=True, help='说话人嵌入目录（每个子目录一个 .pkl）')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--max_speakers', type=int, default=0, help='最大说话人数（0=全部，用于快速测试）')
    parser.add_argument('--top_k', type=int, default=100, help='每个说话人记录的最相似 K 个其他说话人')
    parser.add_argument('--num_processes', type=int, default=0, help='进程数（0=自动）')
    parser.add_argument('--batch_size', type=int, default=200, help='分块处理大小')
    args = parser.parse_args()

    if args.num_processes <= 0:
        args.num_processes = cpu_count()
    print(f'使用 {args.num_processes} 个进程')

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'从以下路径加载说话人嵌入: {args.embeddings_dir}')
    emb_matrix, speaker_ids, _ = load_speaker_embeddings(args.embeddings_dir)
    n = emb_matrix.shape[0]
    print(f'已加载 {n} 个说话人嵌入, 维度: {emb_matrix.shape[1]}')

    if args.max_speakers > 0 and args.max_speakers < n:
        rng = np.random.RandomState(42)
        indices = rng.choice(n, size=args.max_speakers, replace=False)
        emb_matrix = emb_matrix[indices]
        speaker_ids = [speaker_ids[i] for i in indices]
        n = emb_matrix.shape[0]
        print(f'限制为 {n} 个说话人（随机采样）')

    n_pairs = int(n * (n - 1) / 2)
    print(f'唯一互说人配对总数: {n_pairs:,}')

    # === Step 1: Per-speaker top-K most similar speakers ===
    chunks = list(range(n))
    chunk_list = [chunks[i:i+args.batch_size] for i in range(0, n, args.batch_size)]
    task_args = [(c, emb_matrix, args.top_k) for c in chunk_list]

    all_results = []
    with Pool(processes=args.num_processes) as pool:
        for batch_results in tqdm(pool.imap_unordered(compute_chunk_top_sims, task_args),
                                   total=len(task_args), desc='计算 top-K', ncols=100):
            all_results.extend(batch_results)

    speaker_top = {}
    for r in all_results:
        i = r['query_idx']
        speaker_top[speaker_ids[i]] = [
            {'speaker_id': speaker_ids[t['idx']], 'similarity': t['similarity']}
            for t in r['top_k']
        ]

    top_path = os.path.join(args.output_dir, 'speaker_top_similarities.json')
    with open(top_path, 'w') as f:
        json.dump(speaker_top, f, ensure_ascii=False, indent=2)
    print(f'已保存 Top 相似度: {top_path}')

    # === Step 2: Global distribution via random sampling ===
    sample_size = min(n_pairs, 50_000_000)
    print(f'采样 {sample_size:,} 对进行全局统计...')

    rng = np.random.RandomState(42)
    stat_sims = np.empty(sample_size, dtype=np.float32)
    pos = 0
    while pos < sample_size:
        i = rng.randint(0, n)
        j = rng.randint(0, n)
        if i >= j:
            continue
        stat_sims[pos] = emb_matrix[i] @ emb_matrix[j]
        pos += 1

    sampled_sims = stat_sims.astype(np.float64)

    # histogram
    hist, bin_edges = np.histogram(sampled_sims, bins=200, range=(0.0, 1.0))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # percentile stats
    percentiles = np.percentile(sampled_sims, [1, 5, 10, 25, 50, 75, 90, 95, 99])

    stats = {
        'total_speakers': int(n),
        'total_pairs': int(n_pairs),
        'sampled_pairs': int(sample_size),
        'mean_similarity': round(float(np.mean(sampled_sims)), 6),
        'std_similarity': round(float(np.std(sampled_sims)), 6),
        'min_similarity': round(float(np.min(sampled_sims)), 6),
        'max_similarity': round(float(np.max(sampled_sims)), 6),
        'median_similarity': round(float(np.median(sampled_sims)), 6),
        'percentile_1': round(float(percentiles[0]), 6),
        'percentile_5': round(float(percentiles[1]), 6),
        'percentile_10': round(float(percentiles[2]), 6),
        'percentile_25': round(float(percentiles[3]), 6),
        'percentile_75': round(float(percentiles[5]), 6),
        'percentile_90': round(float(percentiles[6]), 6),
        'percentile_95': round(float(percentiles[7]), 6),
        'percentile_99': round(float(percentiles[8]), 6),
        'histogram': hist.astype(int).tolist(),
        'bin_edges': [round(float(e), 4) for e in bin_edges],
        'bin_centers': [round(float(c), 4) for c in bin_centers],
    }

    stats_path = os.path.join(args.output_dir, 'inter_speaker_summary.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f'已保存摘要: {stats_path}')

    # Save full sampled similarities for histogram plotting
    hist_out = {
        'sampled_similarities': sampled_sims.astype(np.float32),
        'histogram': hist.tolist(),
        'bin_edges': bin_edges.tolist(),
    }
    hist_path = os.path.join(args.output_dir, 'inter_speaker_distribution.pkl')
    with open(hist_path, 'wb') as f:
        pickle.dump(hist_out, f)

    print(f'\n===== 说话人间相似度摘要 =====')
    print(f'说话人总数: {n}')
    print(f'平均相似度: {stats["mean_similarity"]:.4f}')
    print(f'标准差: {stats["std_similarity"]:.4f}')
    print(f'中位数: {stats["median_similarity"]:.4f}')
    print(f'范围: [{stats["min_similarity"]:.4f}, {stats["max_similarity"]:.4f}]')
    print(f'百分位: P1={stats["percentile_1"]:.4f}  P5={stats["percentile_5"]:.4f}  '
          f'P25={stats["percentile_25"]:.4f}  P75={stats["percentile_75"]:.4f}  '
          f'P95={stats["percentile_95"]:.4f}  P99={stats["percentile_99"]:.4f}')


if __name__ == '__main__':
    main()
