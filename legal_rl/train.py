import argparse
import json
from pathlib import Path
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from tqdm import trange
from .common import *


def sequence_logprob(model, tok, prefix, completion):
    pre = tok.encode(prefix, add_special_tokens=False)
    suffix = tok.encode(completion, add_special_tokens=False)
    if not suffix:
        return None
    ids = torch.tensor([pre + suffix], device=model.device)
    logits = model(ids).logits[0, len(pre)-1:-1].float()
    targets = torch.tensor(suffix, device=model.device)
    return F.log_softmax(logits, -1).gather(1, targets[:, None]).squeeze(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=100); p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=256); p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--lr', type=float, default=1e-5); p.add_argument('--clip', type=float, default=0.2)
    p.add_argument('--epochs-per-group', type=int, default=1); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--branch', action='store_true', help='Use branch_prefix when present; loss applies only to generated suffix')
    a = p.parse_args(); seed_all(a.seed)
    rows = read_jsonl(a.data); model, tok = load_model(a.model, train=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                              target_modules='all-linear', task_type='CAUSAL_LM'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr)
    Path(a.output).mkdir(parents=True, exist_ok=True)
    with open(Path(a.output) / 'train.jsonl', 'w', encoding='utf-8') as log:
        for step in trange(a.steps):
            row = random.choice(rows)
            fixed = row.get('branch_prefix', '') if a.branch else ''
            prefix = prompt(row, tok) + fixed
            model.eval()
            completions = generate(model, tok, prefix, a.group_size, a.max_new_tokens, a.temperature)
            rewards = [float(parse(fixed + c, row['labels'])[1] == row['answer']) for c in completions]
            advantages = (np.array(rewards) - np.mean(rewards)) / (np.std(rewards) + 1e-6)
            with torch.no_grad():
                old = [sequence_logprob(model, tok, prefix, c) for c in completions]
            model.train(); loss_value = 0.0
            for _ in range(a.epochs_per_group):
                optimizer.zero_grad()
                for completion, advantage, old_lp in zip(completions, advantages, old):
                    if old_lp is None: continue
                    current = sequence_logprob(model, tok, prefix, completion)
                    ratio = (current - old_lp).exp()
                    adv = torch.tensor(advantage, device=model.device, dtype=current.dtype)
                    loss = -torch.minimum(ratio * adv, ratio.clamp(1-a.clip, 1+a.clip) * adv).mean() / a.group_size
                    loss.backward(); loss_value += loss.item()
                optimizer.step()
            log.write(json.dumps({'step': step, 'id': row['id'], 'rewards': rewards,
                                  'success_rate': float(np.mean(rewards)), 'loss': loss_value,
                                  'generated_tokens': sum(len(tok.encode(c, add_special_tokens=False)) for c in completions)}) + '\n')
            log.flush()
    model.save_pretrained(a.output); tok.save_pretrained(a.output)


if __name__ == '__main__': main()
