# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# Original Copyright huggingface and its affiliates. Licensed under the Apache-2.0 License as part
# of huggingface's transformers package.
# Credit https://github.com/huggingface/transformers/blob/master/examples/run_glue.py

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import glue_convert_examples_to_features as convert_examples_to_features

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import os
current_file_path = __file__
current_dir = os.path.dirname(current_file_path)
bert_base_uncased = f'{current_dir}/bert-base-uncased'
modernbert_base = f'{current_dir}/ModernBERT-base'
modernbert_large = f'{current_dir}/ModernBERT-large'


class Classifier:
    def __init__(self, label_list, device, cache_dir, model_type='bert-base-uncased'):
        self._label_list = label_list
        self._device = device

        print(model_type)

        if model_type == 'bert-base-uncased':
            BERT_MODEL = bert_base_uncased
        elif model_type == 'ModernBERT-base':  # v4.48.0 of transformers
            BERT_MODEL = modernbert_base
        elif model_type == 'ModernBERT-large':
            BERT_MODEL = modernbert_large
        else:
            raise ValueError(f'Invalid model type: {model_type}')
                
        self._tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True, use_fast=False)
        self._model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=len(label_list))

        self._model.to(device)

        self._optimizer = None

        self._dataset = {}
        self._data_loader = {}

    def load_data(self, set_type, examples, batch_size, max_length, shuffle):
        self._dataset[set_type] = examples
        self._data_loader[set_type] = _make_data_loader(
            examples=examples,
            label_list=self._label_list,
            tokenizer=self._tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            shuffle=shuffle)

    def get_optimizer(self, learning_rate, warmup_steps, t_total):
        self._optimizer, self._scheduler = _get_optimizer(
            self._model, learning_rate=learning_rate,
            warmup_steps=warmup_steps, t_total=t_total)

    def train_epoch(self):
        self._model.train()

        for step, batch in enumerate(tqdm(self._data_loader['train'],
                                          desc='Training')):
            batch = tuple(t.to(self._device) for t in batch)
            inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'labels': batch[3]}

            self._optimizer.zero_grad()
            outputs = self._model(**inputs)
            loss = outputs[0]  # model
            loss.backward()
            self._optimizer.step()
            self._scheduler.step()

    def evaluate(self, set_type):
        self._model.eval()

        preds_all, labels_all = [], []
        data_loader = self._data_loader[set_type]

        for batch in tqdm(data_loader, desc="Evaluating {} set".format(set_type)):
            batch = tuple(t.to(self._device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'labels': batch[3]}

            with torch.no_grad():
                outputs = self._model(**inputs)
                tmp_eval_loss, logits = outputs[:2]
            preds = torch.argmax(logits, dim=1)

            preds_all.append(preds)
            labels_all.append(inputs["labels"])

        preds_all = torch.cat(preds_all, dim=0)
        labels_all = torch.cat(labels_all, dim=0)

        simple_acc = torch.sum(preds_all == labels_all).item() / labels_all.shape[0]

        acc = accuracy_score(labels_all.cpu(), preds_all.cpu())
        f1_micro = f1_score(labels_all.cpu(), preds_all.cpu(), average='micro')  # average: None | Literal["micro", "macro", "samples", "weighted", "binary", "binary"] = "binary",
        f1_macro = f1_score(labels_all.cpu(), preds_all.cpu(), average='macro')
        f1_weighted = f1_score(labels_all.cpu(), preds_all.cpu(), average='weighted')

        results = {
            "acc": simple_acc,
            "accuracy": acc,
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }

        return results


def _get_optimizer(model, learning_rate, warmup_steps, t_total):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    return optimizer, scheduler


def _make_data_loader(examples, label_list, tokenizer, batch_size, max_length, shuffle):
    # features = convert_examples_to_features(examples,
    #                                         tokenizer,
    #                                         label_list=label_list,
    #                                         max_length=max_length,
    #                                         output_mode="classification")

    # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    # all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    # all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    # all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    # dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    # return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


    features = []
    for example in examples:
        input_ids = tokenizer(example.text_a, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
        label = torch.tensor([label_list.index(example.label)], dtype=torch.long)
        features.append((input_ids['input_ids'], input_ids['attention_mask'], input_ids['token_type_ids'], label))
    
    input_ids, attention_masks, token_type_ids, labels = zip(*features)
    input_ids = torch.cat(input_ids, dim=0).view(-1, max_length)
    attention_masks = torch.cat(attention_masks, dim=0).view(-1, max_length)
    token_type_ids = torch.cat(token_type_ids, dim=0).view(-1, max_length)
    labels = torch.cat(labels, dim=0)
    
    dataset = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
