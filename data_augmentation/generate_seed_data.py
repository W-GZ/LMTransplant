import random
import os
import json
import shutil
from utils import json_jsonl_read, json_jsonl_write, DATASET_METATYPES



def dataset_subsample_class_balanced(source_file, target_file, pre_class_num, exp_id):
    random.seed(exp_id)

    source_file = json_jsonl_read(source_file)

    # Group by category
    grouped_data = {}
    for item in source_file:
        category = item["label"]
        text = item["text"]

        if category not in grouped_data:
            grouped_data[category] = []
        grouped_data[category].append(item)

    # Randomly extract pre_class_num samples for each category
    subsampled_data = []
    
    for category, items in grouped_data.items():
        if pre_class_num < len(items):
            selected_samples = random.sample(items, pre_class_num)
        else:
            selected_samples = items

        subsampled_data.extend(selected_samples)
    
    json_jsonl_write(target_file, subsampled_data)


def dataset_subsample(source_file, target_file, subsample_num, exp_id):
    random.seed(exp_id)

    data_ = json_jsonl_read(source_file)
    subsampled_data = random.sample(data_, min(subsample_num, len(data_)))
    json_jsonl_write(target_file, subsampled_data)



def get_seed_data(dataset_name, exp_id, subsample_num):
    task_name = DATASET_METATYPES[dataset_name]["task_type"]

    if task_name == 'classification':
        pre_class_num = int(subsample_num / len(DATASET_METATYPES[dataset_name]["label_set"]))
        save_path = f'./datasets/{task_name}/{dataset_name}/data_subsample/exp_{exp_id:02}'

        train_data_path = f'./datasets/{task_name}/{dataset_name}/data_jsonl/train.jsonl'
        target_file = f'{save_path}/train_subsample_{subsample_num:03}.jsonl'
        dataset_subsample_class_balanced(train_data_path, target_file, pre_class_num, exp_id)

        dev_data_path = f'./datasets/{task_name}/{dataset_name}/data_jsonl/dev.jsonl'
        target_file = f'{save_path}/dev_subsample_{subsample_num:03}.jsonl'
        dataset_subsample_class_balanced(dev_data_path, target_file, pre_class_num, exp_id)
        
    elif task_name == 'qa':
        save_path = f'./datasets/{task_name}/{dataset_name}/data_subsample/exp_{exp_id:02}'

        train_data_path = f'./datasets/{task_name}/{dataset_name}/data_jsonl/train.jsonl'
        target_file = f'{save_path}/train_subsample_{subsample_num:03}.jsonl'
        dataset_subsample(train_data_path, target_file, subsample_num, exp_id)

        dev_data_path = f'./datasets/{task_name}/{dataset_name}/data_jsonl/dev.jsonl'
        target_file = f'{save_path}/dev_subsample_{subsample_num:03}.jsonl'
        dataset_subsample(dev_data_path, target_file, subsample_num, exp_id)



if __name__ == "__main__":
    """
    generate seed samples in ./datasets/{task_name}/{dataset_name}/data_subsample/exp_00/
    """

    '''class balanced, randomly subsample 10/15/20 examples per class, as LLM2LLM, ZeroShotDataAug, AugGPT'''
    all_datasets = ['SST2']

    subsample_nums = {
        'SST2': [20],
    }

    for dataset_name in all_datasets:
        for exp_id in range(10):
            for subsample_num in subsample_nums[dataset_name]:
                get_seed_data(dataset_name, exp_id, subsample_num)

