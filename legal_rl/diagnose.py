import argparse
import pickle
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from .common import *


def node_prefixes(p, completion):
    result = {'start': p}
    for slot in SLOTS:
        match = re.search(rf'\[{slot}\][^\r\n]*(?:\r?\n|$)', completion, re.I)
        if match: result[slot] = p + completion[:match.end()]
    result['end'] = final_prefix(p, completion)
    return result


@torch.no_grad()
def hidden(model, tok, text):
    ids = tok(text, return_tensors='pt', add_special_tokens=False).to(model.device)
    return model(**ids, output_hidden_states=True).hidden_states[-1][0, -1].float().cpu().numpy()


@torch.no_grad()
def ccr(model, tok, p, completion, labels):
    before_final = re.split(r'\[final\]', completion, maxsplit=1, flags=re.I)[0]
    prompt_ids = tok.encode(p, add_special_tokens=False)
    reasoning_ids = tok.encode(before_final, add_special_tokens=False)
    final_ids = tok.encode('[final]', add_special_tokens=False)
    prefix_ids = prompt_ids + reasoning_ids + final_ids
    full_scores, masked_scores = [], []
    for label in labels:
        label_ids = tok.encode(' ' + label, add_special_tokens=False)
        ids = torch.tensor([prefix_ids + label_ids], device=model.device)
        length = ids.shape[1]
        full = model(ids).logits[0, len(prefix_ids)-1:-1].float().log_softmax(-1)
        mask = torch.full((1, 1, length, length), torch.finfo(model.dtype).min, dtype=model.dtype, device=model.device)
        mask[0, 0] = torch.triu(mask[0, 0], diagonal=1)
        # Block final marker and answer positions from attending to all reasoning tokens.
        mask[:, :, len(prompt_ids)+len(reasoning_ids):, len(prompt_ids):len(prompt_ids)+len(reasoning_ids)] = torch.finfo(model.dtype).min
        blocked = model(ids, attention_mask=mask).logits[0, len(prefix_ids)-1:-1].float().log_softmax(-1)
        target = torch.tensor(label_ids, device=model.device)
        indices = torch.arange(len(target), device=model.device)
        full_scores.append(full[indices, target].sum()); masked_scores.append(blocked[indices, target].sum())
    lp = torch.stack(full_scores).log_softmax(0); lq = torch.stack(masked_scores).log_softmax(0)
    return float((lp.exp() * (lp-lq)).sum().cpu())


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--adapter'); p.add_argument('--rollouts', required=True)
    p.add_argument('--output', required=True); p.add_argument('--fit-probe', help='Save probe fitted on these rollouts; use separate data to evaluate')
    p.add_argument('--probe', help='Probe fitted on a disjoint split from the same model')
    a = p.parse_args()
    if bool(a.fit_probe) == bool(a.probe): p.error('Specify exactly one of --fit-probe or --probe')
    model, tok = load_model(a.model, a.adapter); model.config._attn_implementation = 'eager'
    rows = read_jsonl(a.rollouts); vectors = []
    for row in tqdm(rows):
        nodes = node_prefixes(prompt(row, tok), row['completion'])
        vectors.append({name: hidden(model, tok, text) for name, text in nodes.items()})
    if a.fit_probe:
        # A separate classifier per task avoids mixing incompatible label sets.
        probes = {}
        for task in set(r['task'] for r in rows):
            indices = [i for i, r in enumerate(rows) if r['task'] == task and r['prediction'] is not None]
            x = [vectors[i][node] for i in indices for node in vectors[i]]
            y = [rows[i]['prediction'] for i in indices for node in vectors[i]]
            probes[task] = LogisticRegression(max_iter=1000).fit(x, y)
        with open(a.fit_probe, 'wb') as f: pickle.dump(probes, f)
        write_jsonl(a.output, [])
        return
    with open(a.probe, 'rb') as f: probes = pickle.load(f)
    out = []
    for row, vec in tqdm(list(zip(rows, vectors))):
        probe = probes[row['task']]; prediction = row['prediction']
        if prediction is None or prediction not in probe.classes_: continue
        index = list(probe.classes_).index(prediction)
        confidence = {node: float(probe.predict_proba(v[None])[0, index]) for node, v in vec.items()}
        positions = [s for s in SLOTS if s in confidence]
        previous = confidence['start']; jumps = {}
        for slot in positions:
            jumps[slot] = abs(confidence[slot] - previous); previous = confidence[slot]
        out.append({**row, 'probe_confidence': confidence, 'ECI': confidence['start'] / max(confidence['end'], 1e-8),
                    'CCR': ccr(model, tok, prompt(row, tok), row['completion'], row['labels']),
                    'bottleneck': max(jumps, key=jumps.get) if jumps else None})
    write_jsonl(a.output, out)


if __name__ == '__main__': main()
