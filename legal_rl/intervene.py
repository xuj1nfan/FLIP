import argparse
import re
from tqdm import tqdm
from .common import *


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--adapter')
    p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    p.add_argument('--slot', choices=SLOTS, help='Target or control slot; defaults to per-row bottleneck')
    p.add_argument('--max-new-tokens', type=int, default=256); p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(); seed_all(a.seed); model, tok = load_model(a.model, a.adapter); out = []
    for row in tqdm(read_jsonl(a.data)):
        slot = a.slot or row.get('bottleneck')
        if not slot: continue
        pfx = prompt(row, tok); flipped = flip_prefix(pfx, row['completion'], slot)
        if flipped is None: continue
        continuation = generate(model, tok, flipped, max_new_tokens=a.max_new_tokens, temperature=0)[0]
        original_prefix = final_prefix(pfx, row['completion'])
        altered_completion = flipped[len(pfx):] + continuation
        altered_prefix = final_prefix(pfx, altered_completion)
        with torch.no_grad():
            original = label_logprobs(model, tok, original_prefix, row['labels']).exp().cpu().numpy()
            changed = label_logprobs(model, tok, altered_prefix, row['labels']).exp().cpu().numpy()
        _, prediction = parse(altered_completion, row['labels'])
        gold = row['labels'].index(row['answer'])
        # C: gold probability change, E: total distribution shift, Q: answer flip.
        out.append({**row, 'intervention_slot': slot, 'flipped_completion': altered_completion,
                    'flipped_prediction': prediction, 'C': float(abs(changed[gold]-original[gold])),
                    'E': float(0.5 * abs(changed-original).sum()),
                    'Q': float(prediction != row['prediction']) if prediction is not None and row['prediction'] is not None else None})
    write_jsonl(a.output, out)


if __name__ == '__main__': main()
