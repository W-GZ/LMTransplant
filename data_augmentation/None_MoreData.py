import random
import os
import json
import shutil
from utils import json_jsonl_read, json_jsonl_write, DATASET_METATYPES, dataset2text_label



def get_data(dataset_name, train_data, subsampled_data, augmentation_num, exp_id, subsample_num, file_name_moredata):
    task_name = DATASET_METATYPES[dataset_name]["task_type"]
    random.seed(exp_id)

    if task_name == 'classification':  # dataset_subsample_more_data_class_balanced
        pre_class_num = int(subsample_num / len(DATASET_METATYPES[dataset_name]["label_set"]))

        existing_data = set(item["text"] for item in subsampled_data)
        print(len(existing_data), subsample_num)

        grouped_data = {}
        for item in train_data:
            category = item["label"]
            if category not in grouped_data:
                grouped_data[category] = []
            grouped_data[category].append(item)

        more_data = []
        for category, items in grouped_data.items():
            category_data = []

            if len(items) < pre_class_num * (augmentation_num + 1):
                print("类别数据量不足", category, len(items), pre_class_num * augmentation_num)
                return

            while len(category_data) < pre_class_num * augmentation_num:
                random_data = random.choice(items)
                if random_data["text"] not in existing_data:
                    category_data.append(random_data)
                    more_data.append(random_data)
                    existing_data.add(random_data["text"])

        combined_data = [item for item in subsampled_data] + more_data

        data_augmentation_path = f'./datasets/{task_name}/{dataset_name}/data_augmentation/augmentation_num_{augmentation_num:02}/exp_{exp_id:02}/MoreData'
        json_jsonl_write(os.path.join(data_augmentation_path, file_name_moredata), combined_data)

    else:
        print("This dataset is not a classification dataset")
        existing_data = set(item[dataset2text_label[dataset_name]["text"]] for item in subsampled_data)

        print(len(existing_data), subsample_num)

        more_data = []
        while len(more_data) < min(subsample_num * augmentation_num, len(train_data)):
            random_data = random.choice(train_data)
            if random_data[dataset2text_label[dataset_name]["text"]] not in existing_data:
                more_data.append(random_data)
                existing_data.add(random_data[dataset2text_label[dataset_name]["text"]])

        random.shuffle(more_data)
        combined_data = [item for item in subsampled_data] + more_data
        
        data_augmentation_path = f'./datasets/{task_name}/{dataset_name}/data_augmentation/augmentation_num_{augmentation_num:02}/exp_{exp_id:02}/MoreData'
        json_jsonl_write(os.path.join(data_augmentation_path, file_name_moredata), combined_data)


if __name__ == "__main__":
    augmentation_num = 3
    all_datasets = ['SST2']

    subsample_nums = {
        'SST2': [20],
    }

    for dataset_name in all_datasets:
        task_name = DATASET_METATYPES[dataset_name]["task_type"]
        for exp_id in range(10):
            for subsample_num in subsample_nums[dataset_name]:
                train_data_path = f'./datasets/{task_name}/{dataset_name}/data_jsonl/train.jsonl'
                train_data = json_jsonl_read(train_data_path)

                subsampled_data_path = f'./datasets/{task_name}/{dataset_name}/data_subsample/exp_{exp_id:02}/train_subsample_{subsample_num:03}.jsonl'
                subsampled_data = json_jsonl_read(subsampled_data_path)

                # copy to data_augmentation/none/
                file_name_none = 'train_subsample_{:03}_none.jsonl'.format(subsample_num)
                data_augmentation_path = f'./datasets/{task_name}/{dataset_name}/data_augmentation/augmentation_num_{augmentation_num:02}/exp_{exp_id:02}/none'
                os.makedirs(data_augmentation_path, exist_ok=True)
                shutil.copy(subsampled_data_path, os.path.join(data_augmentation_path, file_name_none))

                # more data, data_augmentation/MoreData/
                file_name_moredata = 'train_subsample_{:03}_MoreData.jsonl'.format(subsample_num)
                get_data(dataset_name, train_data, subsampled_data, augmentation_num, exp_id, subsample_num, file_name_moredata)

