import argparse
from tqdm import tqdm
from .common import *


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--adapter')
    p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    p.add_argument('--samples', type=int, default=1); p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--max-new-tokens', type=int, default=256); p.add_argument('--seed', type=int, default=42)
    a = p.parse_args(); seed_all(a.seed)
    model, tok = load_model(a.model, a.adapter)
    rows = []
    for item in tqdm(read_jsonl(a.data)):
        prefix = prompt(item, tok)
        for i, completion in enumerate(generate(model, tok, prefix, a.samples, a.max_new_tokens, a.temperature)):
            slots, answer = parse(completion, item['labels'])
            rows.append({**item, 'sample': i, 'completion': completion, 'slots': slots,
                         'prediction': answer, 'correct': answer == item['answer']})
    write_jsonl(a.output, rows)


if __name__ == '__main__': main()
