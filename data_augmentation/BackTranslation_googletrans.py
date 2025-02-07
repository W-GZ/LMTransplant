import argparse
import os
import json
import random
from tqdm import tqdm
from multiprocessing import Process
from utils import json_jsonl_read, json_jsonl_write, ChatGPT, DATASET_METATYPES
from utils import construct_messages_for_get_new_label


"""
ref: ZeroShotDataAug: Generating and Augmenting Training Data with ChatGPT

https://pypi.org/project/googletrans/
https://py-googletrans.readthedocs.io/en/latest/
"""
from googletrans import Translator 


class GoogleTranslator():
    def __init__(self):
        self.translator = Translator()
    
    def trans_from_source_to_target(self, original_text, source_language="en", target_language="zh"):
        # For list of language codes, please refer to https://py-googletrans.readthedocs.io/en/latest/#googletrans-languages

        try:
            translation = self.translator.translate(
                original_text,
                src=source_language,  # If source language is not given, google translate attempts to detect the source language.
                dest=target_language,  # default: en, one of the language codes listed in googletrans.LANGUAGES
            )
            return translation.text
        except Exception as e:
            print(f"Error occurred: {e}")
            return None


def back_translation(original_text, source_language, target_language_list):
    translations = []
    new_originals = []

    google_translator = GoogleTranslator()

    for target_language in target_language_list:
        # translation_text = trans_from_source_to_target(original_text, source_language, target_language)
        
        translation_text = None
        timeout = 0
        while translation_text is None:
            translation_text = google_translator.trans_from_source_to_target(original_text, source_language, target_language)
            timeout += 1
            if timeout > 10 or translation_text is not None:
                break
        
        if translation_text is not None:
            translations.append(translation_text) 

    for temp in zip(translations, target_language_list):
        translation_text = temp[0]
        target_language = temp[1]

        # raw_language_text = trans_from_source_to_target(translation_text, target_language, source_language)

        raw_language_text = None
        timeout = 0
        while raw_language_text is None:
            raw_language_text = google_translator.trans_from_source_to_target(translation_text, target_language, source_language)
            timeout += 1
            if timeout > 10 or raw_language_text is not None:
                break
        
        if raw_language_text is not None:
            new_originals.append(raw_language_text) 

    # # new_originals.append(original_text)

    return translations, new_originals


def main(dataset_name, augmentation_num, exp_id, subsample_num, source_language, target_language_list, augmenter):
    random.seed(exp_id)
    task_name = DATASET_METATYPES[dataset_name]["task_type"]
    
    base_path = f'./datasets/{task_name}/{dataset_name}/data_augmentation/augmentation_num_{augmentation_num:02}/exp_{exp_id:02}/'
    input_path = base_path + f'none/train_subsample_{int(subsample_num):03}_none.jsonl'
    output_path = base_path + f'{augmenter}/train_subsample_{int(subsample_num):03}_{augmenter}.jsonl'
    data = json_jsonl_read(input_path)

    source_language = source_language
    target_language_list = target_language_list[:augmentation_num]

    data_augmentation = []
    for item in tqdm(data, desc='Processing', unit='item'):
        if dataset_name in ["SNIPS", "SST2", "TREC"]:
            input_text = item['text']
            input_label = item['label']

            data_augmentation.append({
                'text': input_text,
                'label': input_label
            })

            input_text = input_text.replace("\n", " ")
            translations, aug_sentences = back_translation(input_text, source_language, target_language_list)

            for aug_sentence in aug_sentences:

                data_augmentation.append({
                    'text': aug_sentence,
                    'label': input_label,
                    "aug_sample_flag": 1
                })

            json_jsonl_write(output_path, data_augmentation)



def generate_sub_process(sub_process_id, dataset_name, augmentation_num, exp_id, subsample_num, augmenter):
    print("子进程执行中>>> pid={},ppid={},编号{}".format(os.getpid(), os.getppid(), sub_process_id))

    source_language = 'en'
    target_language_list = ['ru', 'de', 'ja', 'ko', 'fr', "it", "th", "sv", "es", "ar"]

    main(dataset_name, augmentation_num, exp_id, subsample_num, source_language, target_language_list, augmenter)

    print("子进程终止>>> pid={},ppid={},编号{}".format(os.getpid(), os.getppid(), sub_process_id))


if __name__ == "__main__":
    augmentation_num = 3
    augmenter = "BackTranslation"

    '''Test'''
    # original_text = "翻译源语言, \n可设置为auto"
    # source_language = 'zh-cn'
    # target_language_list = ['ru', 'de', 'ja', 'ko', 'fr', "it", "th", "sv", "es", "ar"]

    # translations, new_originals = back_translation(original_text, source_language, target_language_list)
    # print(translations)
    # print(new_originals)


    '''
    Multi Processing
    python BackTranslation.py
    '''
    print("主进程执行中>>> pid={0}".format(os.getpid()))

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

                sub_process = Process(target=generate_sub_process, name="worker" + str(index), args=(index, dataset_name, augmentation_num, exp_id, subsample_num, augmenter))
                sub_processes.append(sub_process)
                
                index += 1

    for i in range(len(sub_processes)):
        sub_processes[i].start()
    
    for i in range(len(sub_processes)):
        sub_processes[i].join()

    print("主进程终止")






