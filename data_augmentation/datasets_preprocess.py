'''Datasets Preprocess'''
import pandas as pd
import os
import numpy as np
from utils import json_jsonl_write
import csv
from nltk.tokenize import sent_tokenize
from torch.utils.data import random_split


def tsv2jsonl(input_path, save_path):
    jsonl_data = []

    with open(input_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')

        for row in reader:
            label = row[0]
            text = row[1]
            data = {"label": label, "text": text}

            jsonl_data.append(data)

    json_jsonl_write(save_path, jsonl_data)


def MLQA_process(train_data_path, save_path):
    """
    https://huggingface.co/datasets/dkoterwa/mlqa_filtered
    """
    # load
    train_df = pd.read_parquet(train_data_path)
    print(f'MLQA Dataset; Train: {len(train_df)}')  # Train: 41019
    train_data = train_df.to_dict(orient='records')
    # json_jsonl_write(os.path.join(save_path, 'mlqa_filtered.jsonl'), train_data)

    mlqa_filtered_data = train_data

    mlqa_filtered_en = []
    for item in mlqa_filtered_data:
        context = item["context"]
        lang = item["lang"]

        if len(context.split()) <= 80 and len(sent_tokenize(context)) >= 3 and lang == "en":
            mlqa_filtered_en.append({
                "lang": item["lang"],
                "context": item["context"],
                "question": item["question"],
                "answer": item["answer"],
                # "id": item["id"],
            })

    # json_jsonl_write(os.path.join(save_path, 'mlqa_filtered_en.jsonl'), mlqa_filtered_en)
    
    dev_size = int(0.2 * len(mlqa_filtered_en))
    test_size = int(0.2 * len(mlqa_filtered_en))
    train_size = len(mlqa_filtered_en) - test_size - dev_size

    training_data, dev_data, test_data = random_split(mlqa_filtered_en, [train_size, dev_size, test_size])
    json_jsonl_write(os.path.join(save_path, 'train.jsonl'), training_data)
    json_jsonl_write(os.path.join(save_path, 'dev.jsonl'), dev_data)
    json_jsonl_write(os.path.join(save_path, 'test.jsonl'), test_data)


if __name__ == "__main__":
    """ classification datasets preprocess """
    all_datasets = ['SST2']

    for dataset_name in all_datasets:
        input_path = f'./classification/utils/datasets/{dataset_name}'
        save_path = f'./datasets/classification/{dataset_name}/data_jsonl'

        for file_name in ["train", "dev", "test"]:
            tsv2jsonl(os.path.join(input_path, f"{file_name}.tsv"), os.path.join(save_path, f"{file_name}.jsonl"))

    """ MLQA """
    # MLQA_train_data_path = './datasets/qa/MLQA/data_from_huggingface/train-00000-of-00001-acf9fa5759f3dbf3.parquet'
    # MLQA_save_path = './datasets/qa/MLQA/data_jsonl/'
    # MLQA_process(MLQA_train_data_path, MLQA_save_path)

