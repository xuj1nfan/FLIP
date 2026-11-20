import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SLOTS = ("facts", "rule", "exception")
STATES = ("yes", "no", "unknown")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_path, adapter=None, train=False):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None, trust_remote_code=True,
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    model.train(train)
    return model, tokenizer


def prompt(row, tokenizer):
    labels = ", ".join(row["labels"])
    text = ("请根据案情或前提判断，按固定格式逐行输出。状态只能是 yes、no 或 unknown，"
            "每行冒号后简要说明。最终标签必须从给定选项中原样选择。\n"
            f"选项：{labels}\n问题：{row['text']}\n"
            "格式：\n[facts] 状态: 理由\n[rule] 状态: 理由\n"
            "[exception] 状态: 理由\n[final] 标签")
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
    return text + "\n"


def parse(text, labels):
    slots = {}
    for slot in SLOTS:
        match = re.search(rf"\[{slot}\]\s*(yes|no|unknown)\s*[:：]", text, re.I)
        slots[slot] = match.group(1).lower() if match else None
    match = re.search(r"\[final\]\s*([^\r\n]+)", text, re.I)
    answer = match.group(1).strip() if match else None
    return slots, answer if answer in labels else None


@torch.no_grad()
def generate(model, tokenizer, prefix, n=1, max_new_tokens=256, temperature=0.8):
    inputs = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).to(model.device)
    result = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                            temperature=temperature if temperature > 0 else None,
                            num_return_sequences=n, pad_token_id=tokenizer.pad_token_id)
    return [tokenizer.decode(ids[inputs.input_ids.shape[1]:], skip_special_tokens=True) for ids in result]


def label_logprobs(model, tokenizer, prefix, labels):
    """Sequence likelihood of each label after the same final marker."""
    scores = []
    for label in labels:
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        label_ids = tokenizer.encode(" " + label, add_special_tokens=False)
        ids = torch.tensor([prefix_ids + label_ids], device=model.device)
        logits = model(ids).logits[:, len(prefix_ids)-1:-1].float()
        target = torch.tensor(label_ids, device=model.device)
        scores.append(logits.log_softmax(-1)[0, torch.arange(len(target), device=model.device), target].sum())
    return torch.stack(scores).log_softmax(0)


def final_prefix(prompt_text, completion):
    return prompt_text + re.split(r"\[final\]", completion, maxsplit=1, flags=re.I)[0] + "[final]"


def flip_prefix(prompt_text, completion, slot):
    pattern = rf"(\[{slot}\]\s*)(yes|no)(\s*[:：])[^\r\n]*"
    match = re.search(pattern, completion, re.I)
    if not match:
        return None
    flipped = "no" if match.group(2).lower() == "yes" else "yes"
    # Remove the old explanation, which would otherwise contradict the edited state.
    return prompt_text + completion[:match.start()] + match.group(1) + flipped + match.group(3)
