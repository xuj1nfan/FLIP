import argparse
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from .common import read_jsonl, write_jsonl, SLOTS


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('transitions'); s.add_argument('--base', required=True); s.add_argument('--grpo', required=True); s.add_argument('--output', required=True)
    s = sub.add_parser('correlate'); s.add_argument('--data', required=True); s.add_argument('--output', required=True)
    s = sub.add_parser('select'); s.add_argument('--data', required=True); s.add_argument('--output', required=True); s.add_argument('--count', type=int, required=True)
    s.add_argument('--method', choices=['random', 'conventional', 'mechanism'], default='mechanism'); s.add_argument('--eci-threshold', type=float, default=0.8); s.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.command == 'transitions':
        base = {(r['id'], r['sample']): r for r in read_jsonl(a.base)}
        groups = defaultdict(list)
        for r in read_jsonl(a.grpo):
            b = base.get((r['id'], r['sample']))
            if b and not b['correct'] and r['correct']:
                groups[r['task']].append({s: (b['slots'][s] != r['slots'][s]) if b['slots'][s] and r['slots'][s] else None for s in SLOTS})
        out = []
        for task, rows in groups.items():
            for slot in SLOTS:
                values = [r[slot] for r in rows if r[slot] is not None]
                out.append({'task': task, 'slot': slot, 'improved_pairs': len(rows), 'valid_pairs': len(values), 'flip_rate': float(np.mean(values)) if values else None})
        write_jsonl(a.output, out)
    elif a.command == 'correlate':
        rows = read_jsonl(a.data); out = []
        for group, subset in [('all', rows), ('discovery', [r for r in rows if r.get('group') == 'discovery']), ('justification', [r for r in rows if r.get('group') == 'justification'])]:
            for feature in ('C', 'E', 'Q'):
                for target in ('gain', 'alignment'):
                    pairs = [(r[feature], r[target]) for r in subset if r.get(feature) is not None and r.get(target) is not None]
                    rho, pv = spearmanr(*zip(*pairs)) if len(pairs) >= 3 else (float('nan'), float('nan'))
                    out.append({'group': group, 'feature': feature, 'target': target, 'n': len(pairs), 'rho': float(rho) if np.isfinite(rho) else None, 'p': float(pv) if np.isfinite(pv) else None})
        write_jsonl(a.output, out)
    else:
        # Select original questions, not individual sampled trajectories.
        unique = {}
        for row in read_jsonl(a.data):
            previous = unique.get(row['id'])
            if previous is None or (row.get('branch_prefix') and not previous.get('branch_prefix')):
                unique[row['id']] = row
        rows = list(unique.values()); rng = np.random.default_rng(a.seed)
        if a.method == 'random': rng.shuffle(rows)
        else:
            def score(r):
                value = r.get('U', 0) + r.get('alignment', 0)
                if a.method == 'mechanism' and r.get('ECI', 1) < a.eci_threshold:
                    value += r.get('C', 0) + r.get('E', 0) + r.get('Q', 0)
                return value
            rows.sort(key=score, reverse=True)
        write_jsonl(a.output, rows[:a.count])


if __name__ == '__main__': main()
