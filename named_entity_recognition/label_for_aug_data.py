"""
You are a professional named entity recognition (NER) annotation expert. Your task is to tokenize the given sentence, identify the named entities, and assign a corresponding BIO-format label and label ID to each token.
This task includes only the following four types of entities: persons (PER), organizations (ORG), locations (LOC), and miscellaneous names (MISC).
Use the following BIO labels and their corresponding label IDs: {"O": 0, "B-ORG": 1, "B-MISC": 2, "B-PER": 3, "I-PER": 4, "B-LOC": 5, "I-ORG": 6, "I-MISC": 7, "I-LOC": 8}.

Please output the result strictly in the following format (only include these four lines):
sentence: original sentence
entities: ['token1', 'token2', ..., 'tokenN']
labels: [BIO_label1, BIO_label2, ..., BIO_labelN]
IDs: [label_id1, label_id2, ..., label_idN]

Here is an example:
sentence: Camacho took a controversial points decision against the Panamanian in Atlantic City in June in a title fight.
entities: ['Camacho', 'took', 'a', 'controversial', 'points', 'decision', 'against', 'the', 'Panamanian', 'in', 'Atlantic', 'City', 'in', 'June', 'in', 'a', 'title', 'fight', '.']
labels: ['B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
IDs: [3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 5, 8, 0, 0, 0, 0, 0, 0, 0]

Now, please perform named entity recognition and annotation for the following sentence:
{text}

Return only the result. Do not include any explanation or additional content.
sentence:
entities:
labels:
IDs:
"""

import numpy as np
import os
import argparse
from multiprocessing import Process, Queue
from tqdm import tqdm
import random
from utils import json_jsonl_read, json_jsonl_write, ChatGPT, sentences_split_into_parts, DATASET_METATYPES
import time
import re


######### get lable and ids for aug data #########
class LabelGPT():
    def __init__(self, max_length=60):
        self.max_length = max_length
        self.client = ChatGPT()

    def create_single_turn_dialogue(self, example, input_text):
        content = input_text.strip()
        content = ' '.join(content.split()[:int(self.max_length)])

        example_sentence = example['sentence']
        example_entities = example['entities']
        example_labels = example['labels']
        example_ids = example['IDs']

        prompt = (
            f"You are a professional named entity recognition (NER) annotation expert. Your task is to tokenize the given sentence, identify the named entities, and assign a corresponding BIO-format label and label ID to each token.\n"
            f"This task includes only the following four types of entities: persons (PER), organizations (ORG), locations (LOC), and miscellaneous names (MISC).\n"
            "Use the following BIO labels and their corresponding label IDs: {'O': 0, 'B-ORG': 1, 'B-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-LOC': 5, 'I-ORG': 6, 'I-MISC': 7, 'I-LOC': 8}.\n\n"

            f"Please output the result strictly in the following format (only include these four lines):\n"
            f"sentence: original sentence\n"
            f"entities: ['token1', 'token2', ..., 'tokenN']\n"
            f"labels: [BIO_label1, BIO_label2, ..., BIO_labelN]\n"
            f"IDs: [label_id1, label_id2, ..., label_idN]\n\n"

            # f"Here is an example:\n"
            # f"sentence: Camacho took a controversial points decision against the Panamanian in Atlantic City in June in a title fight.\n"
            # f"entities: ['Camacho', 'took', 'a', 'controversial', 'points', 'decision', 'against', 'the', 'Panamanian', 'in', 'Atlantic', 'City', 'in', 'June', 'in', 'a', 'title', 'fight', '.']\n"
            # f"labels: ['B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'O']\n"
            # f"IDs: [3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 5, 8, 0, 0, 0, 0, 0, 0, 0]\n\n"
            f"Here is an example:\n"
            f"sentence: {example_sentence}\n"
            f"entities: {example_entities}\n"
            f"labels: {example_labels}\n"
            f"IDs: {example_ids}\n\n"

            f"Now, please perform named entity recognition and annotation for the following sentence:\n"
            f"{input_text}\n\n"

            f"Return only the result. Do not include any explanation or additional content.\n"
            f"sentence: \n"
            f"entities: \n"
            f"labels: \n"
            f"IDs: \n"
        )

        messages = [
            {'role': 'user', 'content': prompt},
        ]

        return messages


def get_lable_and_ids_for_aug_data(output_result):
    """
    sentence: \n
    entities: \n
    labels: \n
    IDs: \n
    """
    pattern = re.compile(r"(\w+): (.+)")
    matches = pattern.findall(output_result)

    parsed_data = {}
    for key, value in matches:
        if key == "sentence":
            parsed_data[key] = value.strip('"')
        elif key == "entities":
            parsed_data[key] = eval(value)
        elif key == "labels":
            parsed_data[key] = eval(value)
        elif key == "IDs":
            parsed_data[key] = eval(value)

    return parsed_data


######### Main #########
def main(dataset_name, augmentation_num, exp_id, subsample_num, augmenter, input_path=None, output_path=None):
    random.seed(exp_id)

    labelGPT = LabelGPT()
    task_name = DATASET_METATYPES[dataset_name]["task_type"]

    labels2ids = {"O": 0, "B-ORG": 1, "B-MISC": 2, "B-PER": 3, "I-PER": 4, "B-LOC": 5, "I-ORG": 6, "I-MISC": 7,
                  "I-LOC": 8}
    ids2labels = {v: k for k, v in labels2ids.items()}

    if input_path is None and output_path is None:
        base_path = f'../datasets/{task_name}/{dataset_name}/data_augmentation/augmentation_num_{augmentation_num:02}/exp_{exp_id:02}/'
        input_path = base_path + f'{augmenter}/aug_data.jsonl'
        output_path = base_path + f'{augmenter}/train_subsample_{int(subsample_num):03}_{augmenter}.jsonl'
        data = json_jsonl_read(input_path)
    else:  # Test
        data = json_jsonl_read(input_path)[0:5]

    labeled_data = []
    if os.path.exists(output_path):
        labeled_data = json_jsonl_read(output_path)
        start_index = (len(labeled_data) // (augmentation_num + 1)) * (augmentation_num + 1)
        labeled_data = labeled_data[:start_index]
        data = data[start_index:]

    example = {
        "sentence": "Camacho took a controversial points decision against the Panamanian in Atlantic City in June in a title fight.",
        "entities": ['Camacho', 'took', 'a', 'controversial', 'points', 'decision', 'against', 'the', 'Panamanian',
                     'in', 'Atlantic', 'City', 'in', 'June', 'in', 'a', 'title', 'fight', '.'],
        "labels": ['B-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O',
                   'O', 'O'],
        "IDs": [3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 5, 8, 0, 0, 0, 0, 0, 0, 0]
    }
    for item in tqdm(data, desc='Processing', unit='item'):
        if dataset_name == "CONLL2003":
            if "aug_sample_flag" not in item:
                input_text = item['text']
                input_label = item['label']

                labeled_data.append({
                    'text': input_text,
                    'label': input_label
                })
                example = {
                    "sentence": " ".join(input_text),
                    "entities": input_text,
                    "labels": [ids2labels[k] for k in input_label],
                    "IDs": input_label
                }
                # print("##"*30)
                # print(example)

            elif "aug_sample_flag" in item:
                input_text = item['text']

                messages = labelGPT.create_single_turn_dialogue(example, input_text)
                temperature = 0.0
                # print("--"*30)
                # print(messages[0]['content'])

                output_result = labelGPT.client.completions_create(
                    model="deepseek-v3",
                    messages=messages,
                    max_tokens=1024,
                    temperature=temperature
                )
                print("--" * 20)
                print(output_result)

                parsed_data = get_lable_and_ids_for_aug_data(output_result)
                # print("--"*20)
                # print(parsed_data)

                if "sentence" not in parsed_data or "entities" not in parsed_data or "labels" not in parsed_data or "IDs" not in parsed_data:
                    print(f"Error")
                    continue
                else:
                    if len(parsed_data["entities"]) == len(parsed_data["labels"]) == len(parsed_data["IDs"]):
                        sentence = parsed_data['sentence']
                        entities = parsed_data['entities']
                        labels = parsed_data['labels']
                        IDs = parsed_data['IDs']

                        labeled_data.append({
                            'text': entities,
                            'label': IDs,
                            "aug_sample_flag": 1
                        })

            json_jsonl_write(output_path, labeled_data)


if __name__ == "__main__":
    augmentation_num = 3
    '''
    Multi Processing
    python label_for_aug_data.py
    '''
    print("主进程执行中>>> pid={0}".format(os.getpid()))


    def generate_sub_process(sub_process_id, dataset_name, exp_id, subsample_num, augmenter):
        print("子进程执行中>>> pid={},ppid={},编号{}".format(os.getpid(), os.getppid(), sub_process_id))
        main(dataset_name, augmentation_num, exp_id, subsample_num, augmenter)
        print("子进程终止>>> pid={},ppid={},编号{}".format(os.getpid(), os.getppid(), sub_process_id))


    # augmenters = ["EDA"]
    augmenters = ["BackTranslation", "GPT3Mix", "AugGPT"]
    all_datasets = ['CONLL2003']
    subsample_nums = {
        "CONLL2003": [50]
    }

    sub_processes = []
    index = 0
    for augmenter in augmenters:
        for dataset_name in all_datasets:
            for exp_id in range(10):
                for subsample_num in subsample_nums[dataset_name]:
                    # print(f"{augmenter} generation %02i" % index)
                    print(f"{augmenter} generation {index:02}")
                    sub_process = Process(target=generate_sub_process, name="worker" + str(index),
                                          args=(index, dataset_name, exp_id, subsample_num, augmenter))
                    sub_processes.append(sub_process)
                    index += 1

    for i in range(len(sub_processes)):
        sub_processes[i].start()
    for i in range(len(sub_processes)):
        sub_processes[i].join()
    print("主进程终止")

