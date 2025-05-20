import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from transformers import TrainingArguments, Trainer, DataCollatorForTokenClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from utils import json_jsonl_read, json_jsonl_write, DATASET_METATYPES

"""
https://blog.csdn.net/a131529/article/details/134770954
"""


class TokenClassificationDataset(Dataset):
    def __init__(self, datas, tokenizer, max_len=128):
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.datas)

    def ner_tags_to_labels(self, ner_tags, word_ids):
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(ner_tags[word_id])
        return label_ids

    def __getitem__(self, idx):
        text = self.datas[idx]["text"]
        ner_tags = self.datas[idx]["label"]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
            is_split_into_words=True
        )

        word_ids = encoding.word_ids()
        label_ids = self.ner_tags_to_labels(ner_tags, word_ids)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


def compute_metrics(pred):
    labels = pred.label_ids.flatten()
    preds = pred.predictions.argmax(-1).flatten()

    preds = preds[labels != -100]
    labels = labels[labels != -100]

    acc = accuracy_score(labels, preds)
    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    result = {
        "accuracy": acc,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

    # # seqeval
    # result.update(seqeval_result)

    return result


def test_for_CONLL2003():
    labels_dict = {"O": 0, "B-ORG": 1, "B-MISC": 2, "B-PER": 3, "I-PER": 4, "B-LOC": 5, "I-ORG": 6, "I-MISC": 7,
                   "I-LOC": 8}

    train_data_path = "../datasets/ner/CONLL2003/data_augmentation/augmentation_num_03/exp_00/none/train_subsample_050_none.jsonl"
    dev_data_path = "../datasets/ner/CONLL2003/data_subsample/exp_00/dev_subsample_050.jsonl"
    # train_data_path = "../datasets/ner/CONLL2003/data_jsonl/train.jsonl"
    # dev_data_path = "../datasets/ner/CONLL2003/data_jsonl/dev.jsonl"
    test_data_path = "../datasets/ner/CONLL2003/data_jsonl/test.jsonl"

    train_data = json_jsonl_read(train_data_path)
    dev_data = json_jsonl_read(dev_data_path)
    test_data = json_jsonl_read(test_data_path)

    modern_bert_path = "../classification/bert_model/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(modern_bert_path)
    model = AutoModelForTokenClassification.from_pretrained(modern_bert_path, num_labels=len(
        labels_dict))  # ModernBertForTokenClassification
    model.to(device)

    train_dataset = TokenClassificationDataset(train_data, tokenizer)
    dev_dataset = TokenClassificationDataset(dev_data, tokenizer)
    test_dataset = TokenClassificationDataset(test_data, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_on_each_node=True,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")


import random


def main(args, device):
    dataset_name = args.dataset
    augmentation_num = args.aug_num
    experiment_id = args.exp_id
    augmenter = args.augmenter
    seednum = args.seednum
    model_name = args.model_type

    random.seed(experiment_id)
    torch.manual_seed(experiment_id)
    torch.cuda.manual_seed_all(experiment_id)
    torch.backends.cudnn.deterministic = True

    """ data load """
    labels_dict = {"O": 0, "B-ORG": 1, "B-MISC": 2, "B-PER": 3, "I-PER": 4, "B-LOC": 5, "I-ORG": 6, "I-MISC": 7,
                   "I-LOC": 8}

    subsample_name = f'train_subsample_{seednum:03}'
    base_path = f'../datasets/ner/{dataset_name}/data_augmentation/augmentation_num_{augmentation_num:02}/exp_{experiment_id:02}/'
    train_data_path = base_path + f"{augmenter}/{subsample_name}_{augmenter}.jsonl"

    dev_data_path = f"../datasets/ner/{dataset_name}/data_subsample/exp_{experiment_id:02}/dev_subsample_{seednum:03}.jsonl"
    test_data_path = f"../datasets/ner/{dataset_name}/data_jsonl/test.jsonl"

    train_data = json_jsonl_read(train_data_path)
    dev_data = json_jsonl_read(dev_data_path)
    test_data = json_jsonl_read(test_data_path)

    print(f"#train_data: {len(train_data)}")
    print(f"#dev_data: {len(dev_data)}")
    print(f"#test_data: {len(test_data)}")
    print(list(labels_dict.keys()))
    print(model_name)
    print('=' * 60, '\n', 'Training', '\n', '=' * 60, sep='')

    if model_name == "ModernBERT-base":
        BERT_MODEL = "../classification/bert_model/ModernBERT-base"
    elif model_name == "bert-base-uncased":
        BERT_MODEL = "../classification/bert_model/bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(BERT_MODEL, num_labels=len(
        labels_dict))  # ModernBertForTokenClassification
    model.to(device)

    train_dataset = TokenClassificationDataset(train_data, tokenizer)
    dev_dataset = TokenClassificationDataset(dev_data, tokenizer)
    test_dataset = TokenClassificationDataset(test_data, tokenizer)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    model_save_path = f"./save_models_{model_name}/{dataset_name}/augmentation_num_{augmentation_num:02}/exp_{experiment_id:02}/{augmenter}/"
    output_dir = os.path.join(model_save_path, 'results')

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=4e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=8,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_on_each_node=True,
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")
    print('Final Test Acc: {:.4f}, f1_micro: {:.4f}, f1_macro: {:.4f}, f1_weighted: {:.4f}'.format(
        100. * test_results["eval_accuracy"],
        100. * test_results["eval_f1_micro"],
        100. * test_results["eval_f1_macro"],
        100. * test_results["eval_f1_weighted"]
    ))


import os, argparse

if __name__ == '__main__':
    """
    # Baseline classifier
    python train.py --cuda 1 --dataset CONLL2003 --exp_id 0 --augmenter none --seednum 50 --model_type ModernBERT-base > train_subsample_050.log

    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=1, type=int)
    parser.add_argument('--dataset')
    parser.add_argument("--exp_id", default=1, type=int)
    parser.add_argument("--aug_num", default=3, type=int)
    parser.add_argument("--augmenter", default="none", type=str)
    parser.add_argument("--seednum", default=50, type=int)
    # parser.add_argument('--hidden_dropout_prob', default=0.1, type=float)
    # parser.add_argument("--warmup_steps", default=100, type=int,
    #                     help="Linear warmup over warmup_steps.")
    # parser.add_argument("--max_seq_length", default=64, type=int,
    #                     help="The maximum total input sequence length after tokenization. "
    #                          "Sequences longer than this will be truncated, sequences shorter will be padded.")
    # parser.add_argument('--epochs', default=8, type=int)
    # parser.add_argument('--min_epochs', default=0, type=int)
    # parser.add_argument("--learning_rate", default=4e-5, type=float)
    # parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument("--model_type", default="ModernBERT-base", type=str)

    args = parser.parse_args()

    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    main(args, device)


