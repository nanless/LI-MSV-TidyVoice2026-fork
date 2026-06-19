import os
import sys
import argparse
import pickle
import glob
import json
import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def load_speaker_embeddings(speakers_dir, exclude_pattern=None):
    emb_files = sorted(glob.glob(os.path.join(speakers_dir, '**', '*.pkl'), recursive=True))
    if exclude_pattern:
        import fnmatch
        emb_files = [f for f in emb_files if not fnmatch.fnmatch(os.path.basename(f), exclude_pattern)]

    speaker_ids = []
    embeddings = []
    for emb_file in emb_files:
        try:
            with open(emb_file, 'rb') as f:
                emb = pickle.load(f)
            rel_dir = os.path.relpath(os.path.dirname(emb_file), speakers_dir)
            speaker_id = rel_dir
            speaker_ids.append(speaker_id)
            embeddings.append(emb.squeeze().astype(np.float64))
        except Exception as e:
            tqdm.write(f'加载失败 {emb_file}: {e}')

    if len(embeddings) == 0:
        return np.array([]), []

    emb_matrix = np.stack(embeddings, axis=0)
    norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_matrix = emb_matrix / norms

    return emb_matrix, speaker_ids


def compute_chunk_similarities(args_pack):
    chunk_indices, emb_matrix, top_k = args_pack
    n_total = emb_matrix.shape[0]
    results = []

    for i in chunk_indices:
        row = emb_matrix[i:i+1]
        sims = row @ emb_matrix.T
        sims = sims.squeeze(0)

        sorted_indices = np.argsort(sims)[::-1]
        top_sims = []
        for j in sorted_indices:
            if i == j:
                continue
            top_sims.append({'idx': int(j), 'similarity': float(sims[j])})
            if len(top_sims) >= top_k:
                break

        results.append({'query_idx': int(i), 'top_k': top_sims})
    return results


def compute_statistics(emb_matrix, speaker_ids, top_k, num_workers, batch_size):
    n = emb_matrix.shape[0]
    print(f'说话人总数: {n}')

    chunks = list(range(n))
    chunk_list = [chunks[i:i + batch_size] for i in range(0, n, batch_size)]

    task_args = [(c, emb_matrix, top_k) for c in chunk_list]

    all_results = []
    with Pool(processes=num_workers) as pool:
        for batch_results in tqdm(pool.imap_unordered(compute_chunk_similarities, task_args),
                                   total=len(task_args), desc='计算相似度', ncols=100):
            all_results.extend(batch_results)

    speaker_top = {}
    for r in all_results:
        i = r['query_idx']
        speaker_top[speaker_ids[i]] = [
            {'speaker_id': speaker_ids[t['idx']], 'similarity': t['similarity']}
            for t in r['top_k']
        ]

    n_pairs = int(n * (n - 1) / 2)
    print(f'唯一配对总数（上三角）: {n_pairs:,}')

    sample_size = min(n_pairs, 10_000_000)
    if sample_size == 0:
        print('说话人数量不足，无法计算配对统计（需要 >=2）')
        stats = {
            'total_speakers': int(n),
            'total_pairs': 0,
            'sampled_pairs': 0,
            'mean_similarity': 0,
            'std_similarity': 0,
            'min_similarity': 0,
            'max_similarity': 0,
            'median_similarity': 0,
            'threshold_statistics': {},
            'extreme_similarity_pairs': [],
        }
        return {}, stats
    print(f'采样 {sample_size:,} 对进行统计...')

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

    thresholds = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
    threshold_stats = {}
    for th in thresholds:
        count = int(np.sum(sampled_sims >= th))
        pct = count / sample_size * 100
        threshold_stats[str(th)] = {'count_above': int(count), 'percent_above': round(pct, 4)}

    extreme_pairs = []
    seen = set()
    if n >= 2:
        rng2 = np.random.RandomState(84)
        n_pairs_sample = min(2_000_000, int(n * (n - 1) / 2))
        pair_i = np.empty(n_pairs_sample, dtype=np.int32)
        pair_j = np.empty(n_pairs_sample, dtype=np.int32)
        pair_sim = np.empty(n_pairs_sample, dtype=np.float32)
        pos = 0
        while pos < n_pairs_sample:
            ii = rng2.randint(0, n)
            jj = rng2.randint(0, n)
            if ii >= jj:
                continue
            pair_i[pos] = ii
            pair_j[pos] = jj
            pair_sim[pos] = emb_matrix[ii] @ emb_matrix[jj]
            pos += 1
        top_order = np.argsort(pair_sim)[::-1]
        for idx in top_order:
            if len(extreme_pairs) >= 100:
                break
            ii = int(pair_i[idx])
            jj = int(pair_j[idx])
            key = (min(ii, jj), max(ii, jj))
            if key not in seen:
                seen.add(key)
                extreme_pairs.append({
                    'speaker1': speaker_ids[ii],
                    'speaker2': speaker_ids[jj],
                    'similarity': float(pair_sim[idx]),
                })

    stats = {
        'total_speakers': int(n),
        'total_pairs': int(n_pairs),
        'sampled_pairs': int(sample_size),
        'mean_similarity': round(float(np.mean(sampled_sims)), 6),
        'std_similarity': round(float(np.std(sampled_sims)), 6),
        'min_similarity': round(float(np.min(sampled_sims)), 6),
        'max_similarity': round(float(np.max(sampled_sims)), 6),
        'median_similarity': round(float(np.median(sampled_sims)), 6),
        'threshold_statistics': threshold_stats,
        'extreme_similarity_pairs': extreme_pairs,
    }

    return speaker_top, stats


def main():
    parser = argparse.ArgumentParser(description='计算说话人之间的余弦相似度')
    parser.add_argument('--embeddings_dir', required=True, help='嵌入基础目录')
    parser.add_argument('--speakers_subdir', default='embeddings_speakers', help='说话人嵌入子目录（step2 输出）')
    parser.add_argument('--similarities_output_subdir', default='speaker_similarity_analysis', help='相似度输出子目录')
    parser.add_argument('--num_workers', type=int, default=0, help='进程数（0 = 自动）')
    parser.add_argument('--batch_size', type=int, default=100, help='分块处理的 batch 大小')
    parser.add_argument('--top_k', type=int, default=100, help='每个说话人记录的最相似 K 个说话人')
    parser.add_argument('--skip_similarity', action='store_true', help='跳过相似度计算')
    parser.add_argument('--max_speakers', type=int, default=0, help='最大说话人数（0 = 全部）')
    parser.add_argument('--exclude_filename_pattern', default=None, help='排除匹配此模式的文件')
    args = parser.parse_args()

    if args.num_workers <= 0:
        args.num_workers = cpu_count()
    print(f'使用 {args.num_workers} 个进程')

    speakers_full_path = os.path.join(args.embeddings_dir, args.speakers_subdir)
    output_path = os.path.join(args.embeddings_dir, args.similarities_output_subdir)
    os.makedirs(output_path, exist_ok=True)

    if not os.path.isdir(speakers_full_path):
        print(f'错误: 说话人目录不存在: {speakers_full_path}')
        sys.exit(1)

    print(f'从以下路径加载说话人嵌入: {speakers_full_path}')
    emb_matrix, speaker_ids = load_speaker_embeddings(speakers_full_path, args.exclude_filename_pattern)

    if args.max_speakers > 0 and args.max_speakers < len(speaker_ids):
        n = args.max_speakers
        print(f'限制为 {n} 个说话人（随机采样）')
        rng = np.random.RandomState(42)
        indices = rng.choice(len(speaker_ids), size=n, replace=False)
        emb_matrix = emb_matrix[indices]
        speaker_ids = [speaker_ids[i] for i in indices]

    print(f'已加载 {len(speaker_ids)} 个说话人嵌入')
    print(f'嵌入维度: {emb_matrix.shape[1]}')

    speaker_top, stats = compute_statistics(
        emb_matrix, speaker_ids,
        top_k=args.top_k,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    mapping = {str(i): sid for i, sid in enumerate(speaker_ids)}
    mapping_path = os.path.join(output_path, 'speaker_keys_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f'已保存说话人映射: {mapping_path}')

    top_path = os.path.join(output_path, 'speaker_top_similarities.json')
    with open(top_path, 'w') as f:
        json.dump(speaker_top, f, ensure_ascii=False, indent=2)
    print(f'已保存 Top 相似度: {top_path}')

    stats_path = os.path.join(output_path, 'analysis_summary.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f'已保存分析摘要: {stats_path}')

    extreme_path = os.path.join(output_path, 'extreme_similarity_pairs.json')
    with open(extreme_path, 'w') as f:
        json.dump(stats['extreme_similarity_pairs'], f, ensure_ascii=False, indent=2)
    print(f'已保存极端相似度配对: {extreme_path}')

    thresh_path = os.path.join(output_path, 'threshold_statistics.json')
    with open(thresh_path, 'w') as f:
        json.dump(stats['threshold_statistics'], f, ensure_ascii=False, indent=2)
    print(f'已保存阈值统计: {thresh_path}')

    print(f'\n===== 相似度摘要 =====')
    print(f'说话人总数: {stats["total_speakers"]}')
    print(f'平均配对相似度: {stats["mean_similarity"]:.4f}')
    print(f'标准差: {stats["std_similarity"]:.4f}')
    print(f'中位数: {stats["median_similarity"]:.4f}')
    print(f'最小值: {stats["min_similarity"]:.4f}')
    print(f'最大值: {stats["max_similarity"]:.4f}')

    print(f'\n阈值分析:')
    for th, s in stats['threshold_statistics'].items():
        print(f'  >= {th}: {s["count_above"]:,} 对 ({s["percent_above"]:.2f}%)')

    print(f'\n最相似的前 5 对:')
    for pair in stats['extreme_similarity_pairs'][:5]:
        print(f'  {pair["speaker1"]} <-> {pair["speaker2"]}: {pair["similarity"]:.4f}')

    print(f'\n所有结果已保存到: {output_path}')


if __name__ == '__main__':
    main()
