import argparse
import json
import os
import logging
from collections import Counter


dataset2text_label = {
    'SNIPS': {
        'text':'text', 
        'label':'label'
        }, 
    'SST2': {
        'text':'text', 
        'label':'label'
        }, 
    'TREC': {
        'text':'text', 
        'label':'label'
        },
    "MLQA": {
        'text':'question',
        'label':'answer'
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


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument("file", type=str)
    parser.add_argument("N", type=int, help="N in distinct-N metric")
    parser.add_argument("--numbers-only", action="store_true")
    return parser.parse_args()


from collections import Counter
from typing import List

def generate_ngrams(tokens: List[str], n: int) -> List[str]:
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i + n])
        ngrams.append(ngram)
    return ngrams

def calculate_distinct_ngrams(data: List[str], n: int) -> float:
    all_ngrams = []
    for sentence in data:
        words = sentence.split()
        ngrams = generate_ngrams(words, n)
        all_ngrams.extend(ngrams)

    ngram_counts = Counter(all_ngrams)

    # total_ngrams = sum(ngram_counts.values())
    # distinct_ngrams = len(ngram_counts)

    total_ngrams = 0
    distinct_ngrams = 0
    for _, freq in ngram_counts.items():
        total_ngrams += freq
        distinct_ngrams += 1 if freq == 1 else 0
    # print(total_ngrams, distinct_ngrams)

    return distinct_ngrams, total_ngrams


def main():
    args = parse_args()

    results = []
    dataset_name = "SST2"
    augmenter = "none"
    subsample_num = 20

    input_path = f'../datasets/classification/{dataset_name}/data_augmentation/{augmenter}/' + 'train_subsample_{:03}_{}.jsonl'.format(int(subsample_num), augmenter)
    data = json_jsonl_read(input_path)

    examples = []
    print(dataset2text_label[dataset_name]["text"])
    for item in data:
        examples.append(item[dataset2text_label[dataset_name]["text"]])

    n_distinct, n_total = calculate_distinct_ngrams(examples, args.N)
    if not args.numbers_only:
        print(f"distinct {args.N}-grams\ttotal {args.N}-grams\tdistinct proportion")
    print(f"\t{n_distinct}\t{n_total}\t{n_distinct / n_total}")

    results.append({
        "dataset_name": dataset_name,
        "augmenter": augmenter,
        "subsample_num": subsample_num,
        f"distinct {args.N}-grams": n_distinct,
        f"total {args.N}-grams": n_total,
        "distinct proportion": n_distinct / n_total
    })

    output_path = 'result.jsonl'
    json_jsonl_write(output_path, results)


def eval_all():
    all_datasets = [
        'SST2', 
        ]
    subsample_nums = {
        'SST2': [20], 
    }
    augmenters = ["none", "MoreData", "EDA", "BackTranslation", "GPT3Mix", "AugGPT", "LLM2LLM", "ours_l_r", "ours_r_l", "ours_r"]

    
    for dataset_name in all_datasets:
        result_dinstinct_3 = {augmenter: [] for augmenter in augmenters}

        for augmenter in augmenters:
            for i in range(10):
                if dataset_name == "MLQA":
                    path_ = f'../datasets/qa/{dataset_name}/data_augmentation/augmentation_num_03/exp_{i:02}/{augmenter}/'
                else:
                    path_ = f'../datasets/classification/{dataset_name}/data_augmentation/augmentation_num_03/exp_{i:02}/{augmenter}/'

                if not os.path.exists(path_):
                    continue

                data = []
                for subsample_num in subsample_nums[dataset_name]:
                    input_path = path_ + 'train_subsample_{:03}_{}.jsonl'.format(int(subsample_num), augmenter)
                    if os.path.exists(input_path):
                        data += json_jsonl_read(input_path)
                    else:
                        continue
                    
                if len(data) == 0:
                    continue

                examples = []
                for item in data:

                    if dataset_name == "MLQA":
                        if "context" in item:
                            examples.append(item["context"] + " " + item["question"])
                        else:
                            examples.append(item["question"])
                    else:
                        examples.append(item[dataset2text_label[dataset_name]["text"]])


                n_distinct, n_total = calculate_distinct_ngrams(examples, 3)
                result_dinstinct_3[augmenter].append(n_distinct / n_total)


        print("####"*10)
        print(dataset_name)

        """ dinstinct_3 """
        for augmenter in augmenters:
            if len(result_dinstinct_3[augmenter]) == 0:
                continue

            print("--"*5)
            print(augmenter)

            dinstinct_3_temp = result_dinstinct_3[augmenter]
            dinstinct_3_mean = sum(dinstinct_3_temp) / len(dinstinct_3_temp)
            dinstinct_3_variance = sum((x - dinstinct_3_mean) ** 2 for x in dinstinct_3_temp) / len(dinstinct_3_temp)
            dinstinct_3_std_deviation = dinstinct_3_variance ** 0.5

            # print(f"Mean dinstinct_3: {100*dinstinct_3_mean:.2f}({dinstinct_3_std_deviation:.2f})")
            print(f"Mean dinstinct_3: {dinstinct_3_mean:.2f}({dinstinct_3_std_deviation:.2f})")

            print(dinstinct_3_temp)


    # json_jsonl_write('result.json', results)


if __name__ == '__main__':
    # python distinct_n.py 3
    # main()

    eval_all()
