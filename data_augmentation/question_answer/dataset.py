import json
import os
import re
import torch as th


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples_for_dataset(dataset_name, train_data_path):
    if dataset_name == "MLQA":
        return get_examples_for_MLQA(train_data_path)

def get_examples_for_MLQA(path):
    examples = read_jsonl(path)

    for ex in examples:
        if "context" in ex:
            ex.update(question="Question: " + ex["context"] + " " + ex["question"]+ "\nAnswer: ")
            ex.update(answer=ex["answer"] + "<|endoftext|>")
        else:
            ex.update(question="Question: " + ex["question"] + "\nAnswer: ")
            ex.update(answer=ex["answer"] + "<|endoftext|>")

    print("-"*10)
    print(f"{len(examples)} examples")
    
    return examples


def get_examples_with_prompt(path):
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n" + 
                  "Let's think step by step and output the final numeric answer in the format '#### answer':\n")
                
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print("-"*10)
    print(f"{len(examples)} examples")
    
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class GSMDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )
        tokens = th.tensor(tokens)
        mask = th.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask)


"""utils"""

def json_jsonl_read(input_path):
    if input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

    return data


def json_jsonl_write(output_path, data):
    content_before_last_slash = output_path.rsplit('/', 1)[0]
    os.makedirs(content_before_last_slash, exist_ok=True)

    if output_path.endswith('.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')


import torch
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        