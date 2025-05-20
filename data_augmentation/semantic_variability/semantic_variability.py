import argparse
import json
import os
import logging
from collections import Counter

dataset2text_label = {
    'SNIPS': {
        'text': 'text',
        'label': 'label'
    },
    'SST2': {
        'text': 'text',
        'label': 'label'
    },
    'TREC': {
        'text': 'text',
        'label': 'label'
    },
    "CONLL2003": {
        'text': 'text',
        'label': 'label'
    }
}


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

    if 'json' not in content_before_last_slash:
        os.makedirs(content_before_last_slash, exist_ok=True)

    if output_path.endswith('.json'):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')


from bert_score import score
import numpy as np


def eval_all():
    datasets_ = [
        # 'SNIPS',
        # 'SST2',
        # 'TREC',
        # "MLQA",
        "CONLL2003"
    ]
    subsample_nums = {
        'SNIPS': [70],
        'SST2': [20],
        'TREC': [60],
        'MLQA': [50],
        "CONLL2003": [50]
    }
    augmenters = ["EDA",
                  # "BackTranslation",
                  # # "GPT3Mix",
                  # "AugGPT",
                  # # "LLM2LLM",
                  ]

    # per_seed = False
    per_seed = True

    for dataset_name in datasets_:
        print(dataset_name)
        f1_scores = {augmenter: [] for augmenter in augmenters}

        for augmenter in augmenters:
            print(augmenter)
            for i in range(10):
                print(i)
                if dataset_name == "MLQA":
                    path_ = f'../data_augmentation/datasets/{dataset_name}/data_augmentation/augmentation_num_03/exp_{i:02}/{augmenter}/'
                elif dataset_name in ["SST2", "TREC", "SNIPS", "Dengue"]:
                    path_ = f'../data_augmentation/datasets/classification/{dataset_name}/data_augmentation/augmentation_num_03/exp_{i:02}/{augmenter}/'
                elif dataset_name == "CONLL2003":
                    path_ = f'../data_augmentation/datasets/ner/{dataset_name}/data_augmentation/augmentation_num_03/exp_{i:02}/{augmenter}/'
                    # path_ = f'../data_augmentation/datasets/ner_gpt3.5/{dataset_name}/data_augmentation/augmentation_num_03/exp_{i:02}/{augmenter}/'
                    # path_ = f'../data_augmentation/datasets/ner_gpt4o/{dataset_name}/data_augmentation/augmentation_num_03/exp_{i:02}/{augmenter}/'

                if not os.path.exists(path_):
                    continue

                if dataset_name != "CONLL2003":
                    data = []
                    for subsample_num in subsample_nums[dataset_name]:
                        input_path = path_ + 'train_subsample_{:03}_{}.jsonl'.format(int(subsample_num), augmenter)
                        if os.path.exists(input_path):
                            data += json_jsonl_read(input_path)
                        else:
                            continue
                elif dataset_name == "CONLL2003":
                    data = []
                    for subsample_num in subsample_nums[dataset_name]:
                        input_path = path_ + "aug_data.jsonl"
                        if os.path.exists(input_path):
                            data += json_jsonl_read(input_path)
                        else:
                            input_path = path_ + 'train_subsample_{:03}_{}.jsonl'.format(int(subsample_num), augmenter)
                            if os.path.exists(input_path):
                                data += json_jsonl_read(input_path)
                            else:
                                continue

                if len(data) == 0:
                    continue

                original_text = []
                augmented_text = []

                temp_f1 = []

                if dataset_name == "MLQA":
                    for i in range(0, len(data), 4):
                        four_elements = data[i:i + 4]

                        texts = []
                        for element in four_elements:
                            texts.append(element["question"])

                        original_text = texts[:1]
                        augmented_text = texts[1:]
                        original_text = original_text * len(augmented_text)

                        model_type = "../data_augmentation/classification/bert_model/bert-base-uncased"
                        """ way 1: semantic similarity """
                        if not per_seed:
                            P, R, F1 = score(augmented_text, original_text,
                                             lang="en",
                                             model_type=model_type,
                                             num_layers=12,
                                             #  verbose=True
                                             )
                            temp_f1.append(F1.mean().item())

                        """ way 2: per seed """
                        if per_seed:
                            temp_ = []
                            for i in range(len(augmented_text) - 1):
                                for j in range(i + 1, len(augmented_text)):
                                    P, R, F1 = score([augmented_text[i]], [augmented_text[j]],
                                                     lang="en",
                                                     model_type=model_type,
                                                     num_layers=12,
                                                     )
                                    temp_.append(F1.mean().item())
                            temp_f1.append(np.mean(temp_))

                else:
                    for i, item in enumerate(data):
                        temp_text = item[dataset2text_label[dataset_name]["text"]]

                        if "aug_sample_flag" not in item:
                            if dataset_name == "CONLL2003":
                                temp_text = " ".join(temp_text)
                            if len(original_text) == 0:
                                original_text = [temp_text]
                                augmented_text = []
                            else:
                                # Bert Socre
                                original_text = original_text * len(augmented_text)

                                # print(original_text)
                                # print(augmented_text)

                                model_type = "../data_augmentation/classification/bert_model/bert-base-uncased"
                                """ way 1 """
                                if not per_seed:
                                    P, R, F1 = score(augmented_text, original_text,
                                                     lang="en",
                                                     model_type=model_type,
                                                     num_layers=12,
                                                     #  verbose=True
                                                     )
                                    temp_f1.append(F1.mean().item())

                                """ way 2 """
                                if per_seed:
                                    temp_ = []
                                    for i in range(len(augmented_text) - 1):
                                        for j in range(i + 1, len(augmented_text)):
                                            P, R, F1 = score([augmented_text[i]], [augmented_text[j]],
                                                             lang="en",
                                                             model_type=model_type,
                                                             num_layers=12,
                                                             )
                                            temp_.append(F1.mean().item())
                                    temp_f1.append(np.mean(temp_))

                                original_text = [temp_text]
                                augmented_text = []

                        else:
                            augmented_text.append(temp_text)

                f1_scores[augmenter].append(np.mean(temp_f1))

        print("####" * 10)
        print(dataset_name)

        """ f1_scores """
        for augmenter in augmenters:
            if len(f1_scores[augmenter]) == 0:
                continue

            print("--" * 5)
            print(augmenter)

            f1_scores_temp = f1_scores[augmenter]
            f1_scores_mean = sum(f1_scores_temp) / len(f1_scores_temp)
            f1_scores_variance = sum((x - f1_scores_mean) ** 2 for x in f1_scores_temp) / len(f1_scores_temp)
            f1_scores_std_deviation = f1_scores_variance ** 0.5

            # print(f"Mean f1_scores: {100*f1_scores_mean:.2f}({f1_scores_std_deviation:.2f})")
            print(f"Mean f1_scores: {f1_scores_mean:.2f}({f1_scores_std_deviation:.2f})")

            print(f1_scores_temp)

        """ pvalue """
        # pvalues_ = []
        # augmenter1 = "llmda_l_r"
        # dinstinct_3_temp1 = result_dinstinct_3[augmenter1]
        # pvalues = []
        # for augmenter2 in augmenters:
        #     dinstinct_3_temp2 = result_dinstinct_3[augmenter2]

        #     statistic, pvalue = significant_analysis(dinstinct_3_temp1, dinstinct_3_temp2)
        #     pvalues.append(float(pvalue))

        #     print(f"dinstinct_3 {augmenter1} vs {augmenter2}:", pvalue)

        #     # print(f"{augmenter} vs {augmenter}:", np.corrcoef(acc_temp1, acc_temp2)[0, 1])
        # # print(augmenter1)
        # # print(pvalues)
        # # pvalues_.append(pvalues)

        # # print(pvalues_)


if __name__ == '__main__':
    eval_all()

