import argparse
import re
from collections import defaultdict
from .common import *


def main():
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    p.add_argument('--before', help='Baseline rollout JSONL for per-question success gain')
    p.add_argument('--after', help='Post-training rollout JSONL')
    p.add_argument('--eci-threshold', type=float, default=0.8); p.add_argument('--ccr-threshold', type=float, default=0.01)
    a = p.parse_args(); rows = read_jsonl(a.data)
    def rates(path):
        groups = defaultdict(list)
        for r in read_jsonl(path): groups[r['id']].append(float(r['correct']))
        return {k: float(np.mean(v)) for k, v in groups.items()}
    before = rates(a.before) if a.before else {}; after = rates(a.after) if a.after else {}
    out = []
    for row in rows:
        r = dict(row); eci = r.get('ECI'); ccr = r.get('CCR')
        if eci is not None and ccr is not None:
            r['group'] = 'discovery' if eci < a.eci_threshold and ccr >= a.ccr_threshold else ('justification' if eci >= a.eci_threshold and ccr < a.ccr_threshold else 'mixed')
        if r['id'] in before:
            r['U'] = 4 * before[r['id']] * (1-before[r['id']])
            if r['id'] in after: r['gain'] = after[r['id']] - before[r['id']]
        slot = r.get('bottleneck')
        if slot and r.get('correct'):
            match = re.search(rf'\[{slot}\]', r['completion'], re.I)
            if match: r['branch_prefix'] = r['completion'][:match.start()]
        out.append(r)
    write_jsonl(a.output, out)


if __name__ == '__main__': main()
