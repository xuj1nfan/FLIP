"""Cosine alignment of per-example teacher-forced LoRA gradients with validation gradients."""
import argparse
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from .common import *
from .train import sequence_logprob


def gradient(model, tok, row, params):
    model.zero_grad(set_to_none=True)
    lp = sequence_logprob(model, tok, prompt(row, tok), row['completion'])
    if lp is None: return None
    (-lp.mean()).backward()
    return torch.cat([p.grad.detach().float().flatten().cpu() if p.grad is not None else torch.zeros(p.numel()) for p in params])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--model', required=True); p.add_argument('--adapter')
    p.add_argument('--data', required=True); p.add_argument('--validation', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); model, tok = load_model(a.model, a.adapter, train=True)
    if not a.adapter:
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules='all-linear', task_type='CAUSAL_LM'))
    model.eval(); params = [p for p in model.parameters() if p.requires_grad]
    valid = [r for r in read_jsonl(a.validation) if r.get('correct')]
    grads = [gradient(model, tok, r, params) for r in tqdm(valid, desc='validation')]
    reference = torch.stack([g for g in grads if g is not None]).mean(0)
    out = []
    for row in tqdm(read_jsonl(a.data), desc='examples'):
        if not row.get('correct'): continue
        g = gradient(model, tok, row, params)
        if g is None: continue
        alignment = torch.nn.functional.cosine_similarity(g[None], reference[None]).item()
        out.append({**row, 'alignment': alignment})
    write_jsonl(a.output, out)


if __name__ == '__main__': main()
