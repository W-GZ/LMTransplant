import numpy as np
import os
import argparse
from multiprocessing import Process
from tqdm import tqdm
from utils import json_jsonl_read, json_jsonl_write, ChatGPT, sentences_split_into_parts, construct_messages_for_get_new_label, DATASET_METATYPES
import re
import random


######### Our Method #########
class OurMethod():
    def __init__(self, max_length=300):
        self.max_length = max_length
        self.client = ChatGPT()


    ######### Forward Completion #########
    def construct_messages_for_forward_completion(self, sample=None, text_type=None, label_type=None):
        single_turn_prompt = (
            f"Given the original {text_type}, generate a succeeding sentence as follows:\n"
            f"Succeeding Sentence: Generate a sentence that can naturally follow the original {text_type}, ensuring a smooth transition and logical continuation.\n\n"

            f"The original {text_type} is: {sample}\n\n"

            f"Now please return the generated succeeding sentence in the following format:\n"
            f"Original Text: [The original {text_type}]\n"
            f"Succeeding Sentence: [The generated succeeding sentence]\n"
        )

        single_turn_dialogue = [{'role': 'user', 'content': single_turn_prompt}]

        return single_turn_dialogue


    ######### Back Infilling #########
    def _label_enum_str(self, label_set, or_str="or"):
        labels = list(label for label in label_set)
        if len(labels) == 1:
            label_enum_str = labels[0]
        else:
            label_enum_str = (", ".join(map("'{}'".format, labels[:-1])) +
                            f" {or_str} '{labels[-1]}'")
        return label_enum_str

    def construct_messages_for_infilling(self, post_sentence, original_text, text_type=None, label_type=None, original_label=None, label_set=None):
        single_turn_prompt = (
            f"You are provided with two pieces of text:\n"
            f"1. Original {text_type}: {original_text}\n"
            f"2. Succeeding Sentence: {post_sentence}\n\n"

            f"You are an expert in text data augmentation. Your task is to generate a new {text_type} that can replace the original {text_type} while meeting these requirements:\n"
            f"1. Fits naturally in front of succeeding sentence, maintaining logical flow and coherence.\n"
            f"2. Similar in text length, format (sentence pair, etc.), and language style to the original {text_type}.\n"
            f"3. Similar '{label_type}' to the original {text_type}, which is '{original_label}'.\n"
            f"4. The new {text_type} should not simply repeat the original {text_type}.\n\n"

            f"Now please return the generated new {text_type} as 'Middle Sentence' in the following format:\n"
            f"Middle Sentence: [The generated new {text_type}]\n"
            f"Succeeding Sentence: [The provided succeeding sentence]\n"
        )

        single_turn_dialogue = [{'role': 'user', 'content': single_turn_prompt}]

        return single_turn_dialogue
    


def split_text_by_subsentences(text, subsentences):
    result = {}
    for i, subsentence in enumerate(subsentences):
        start_index = text.find(subsentence + ":")

        if start_index != -1:
            start_index += len(subsentence) + 1

            end_index = -1
            for subsentence_ in subsentences[(i + 1):]:
                end_index = text.find(subsentence_ + ":", start_index)

                if end_index != -1:
                    result[subsentence] = text[start_index:end_index].strip()
                    break

            if end_index == -1:
                end_index = len(text)
                result[subsentence] = text[start_index:end_index].strip()

            # print(subsentence, start_index, end_index)

    return result


######### Main #########
def main(dataset_name, augmentation_num, exp_id, subsample_num, augmenter, input_path=None, output_path=None):
    random.seed(exp_id)

    ourmethod = OurMethod()
    task_name = DATASET_METATYPES[dataset_name]["task_type"]

    if input_path is None and output_path is None:
        base_path = f'./datasets/{task_name}/{dataset_name}/data_augmentation/augmentation_num_{augmentation_num:02}/exp_{exp_id:02}/'
        input_path = base_path + f'none/train_subsample_{int(subsample_num):03}_none.jsonl'
        output_path = base_path + f'{augmenter}/train_subsample_{int(subsample_num):03}_{augmenter}.jsonl'
        data = json_jsonl_read(input_path)
    else:  # Test
        data = json_jsonl_read(input_path)[0:5]

    text_type = DATASET_METATYPES[dataset_name]["text_type"]
    label_type = DATASET_METATYPES[dataset_name]["label_type"]

    data_augmentation = []

    # # generate continuously
    text_temp = []

    if os.path.exists(output_path):
        data_augmentation = json_jsonl_read(output_path)

        for item in data_augmentation:
            text_temp.append(item['text'])

    for item in tqdm(data, desc='Processing', unit='item'):
        if dataset_name in ["SNIPS", "SST2", "TREC"]:
            input_text = item['text']
            input_label = item['label']

            if input_text in text_temp:
                continue

            data_augmentation.append({
                'text': input_text,
                'label': input_label
            })

            for temp in range(augmentation_num):
                temperature = float(temp / (augmentation_num-1)) * 1.0

                messages_for_forward_completion = ourmethod.construct_messages_for_forward_completion(sample=input_text, text_type=text_type, label_type=label_type)
                forward_completion = ourmethod.client.completions_create(messages=messages_for_forward_completion, temperature=temperature)
                dict_1 = split_text_by_subsentences(forward_completion, ["Original Text", "Succeeding Sentence"])

                if "Succeeding Sentence" not in dict_1:
                    continue
                else:
                    post_sentence = dict_1['Succeeding Sentence']

                    messages_for_infilling = ourmethod.construct_messages_for_infilling(post_sentence, input_text, text_type=text_type, label_type=label_type, original_label=input_label)
                    text_infilling = ourmethod.client.completions_create(messages=messages_for_infilling, temperature=temperature)
                    dict_2 = split_text_by_subsentences(text_infilling, ["Middle Sentence", "Succeeding Sentence"])


                    if "Middle Sentence" not in dict_2:
                        continue
                    else:
                        new_text = dict_2['Middle Sentence']
                        new_label = input_label

                        data_augmentation.append({
                            'text': new_text,
                            'label': new_label,
                            "middle_sentence": new_text,
                            "post_sentence": post_sentence,
                            "aug_sample_flag": 1,
                        })

            json_jsonl_write(output_path, data_augmentation)



if __name__ == "__main__":
    augmentation_num = 3
    augmenter = 'ours_r'


    print("主进程执行中>>> pid={0}".format(os.getpid()))

    def generate_sub_process(sub_process_id, dataset_name, exp_id, subsample_num):
        print("子进程执行中>>> pid={},ppid={},编号{}".format(os.getpid(), os.getppid(), sub_process_id))
        main(dataset_name, augmentation_num, exp_id, subsample_num, augmenter)
        print("子进程终止>>> pid={},ppid={},编号{}".format(os.getpid(), os.getppid(), sub_process_id))


    all_datasets = ['SST2']
    subsample_nums = {
        'SST2': [20],
    }
    
    sub_processes = []
    index = 0
    for dataset_name in all_datasets:
        for exp_id in range(10):
            for subsample_num in subsample_nums[dataset_name]:

                print("generation %02i" % index)

                sub_process = Process(target=generate_sub_process, name="worker" + str(index), args=(index, dataset_name, exp_id, subsample_num))
                sub_processes.append(sub_process)
                
                index += 1


    for i in range(len(sub_processes)):
        sub_processes[i].start()

    for i in range(len(sub_processes)):
        sub_processes[i].join()

    print("主进程终止")

