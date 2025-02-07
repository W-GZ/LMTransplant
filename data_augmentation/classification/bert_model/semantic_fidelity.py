# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0


import torch, os
import argparse

from semantic_fidelity_data_processors import get_data, InputExample
from bert_model import Classifier
import random


def main(args, device):
    random.seed(args.exp_id)
    torch.manual_seed(args.exp_id)
    torch.cuda.manual_seed_all(args.exp_id)
    torch.backends.cudnn.deterministic = True

    examples, label_list = get_data(args)

    t_total = len(examples['train']) // args.epochs

    classifier = Classifier(label_list=label_list, device=device, cache_dir=args.cache, model_type=args.model_type)
    classifier.get_optimizer(learning_rate=args.learning_rate, warmup_steps=args.warmup_steps, t_total=t_total)

    classifier.load_data('train', examples['train'], args.batch_size, max_length=args.max_seq_length, shuffle=True)
    classifier.load_data('dev', examples['dev'], args.batch_size, max_length=args.max_seq_length, shuffle=False)
    classifier.load_data('test', examples['test'], args.batch_size, max_length=args.max_seq_length, shuffle=False)

    print('=' * 60, '\n', 'Training', '\n', '=' * 60, sep='')
    best_dev_acc, final_test_acc = -1., -1.
    for epoch in range(args.epochs):
        classifier.train_epoch()
        dev_acc = classifier.evaluate('dev')["acc"]

        if epoch >= args.min_epochs:
            do_test = (dev_acc > best_dev_acc)
            best_dev_acc = max(best_dev_acc, dev_acc)
        else:
            do_test = False

        print('Epoch {}, Dev Acc: {:.4f}, Best Ever: {:.4f}'.format(
            epoch, 100. * dev_acc, 100. * best_dev_acc))

        if do_test:
            results = classifier.evaluate('test')
            final_test_acc = results["acc"]

            accuracy = results["accuracy"]
            f1_micro = results["f1_micro"]
            f1_macro = results["f1_macro"]
            f1_weighted = results["f1_weighted"]
            print('Test Acc: {:.4f}  accuracy: {:.4f}  f1_micro: {:.4f}  f1_macro: {:.4f}  f1_weighted: {:.4f}'.format(
                    100. * final_test_acc, 
                    100. * accuracy, 
                    100. * f1_micro,
                    100. * f1_macro, 
                    100. * f1_weighted))

    # print('Final Dev Acc: {:.4f}, Final Test Acc: {:.4f}'.format(100. * best_dev_acc, 100. * final_test_acc))
    print('Final Dev Acc: {:.4f}, Final Test Acc: {:.4f}, accuracy: {:.4f}, f1_micro: {:.4f}, f1_macro: {:.4f}, f1_weighted: {:.4f}'.format(
        100. * best_dev_acc, 
        100. * final_test_acc,
        100. * accuracy,
        100. * f1_micro,
        100. * f1_macro,
        100. * f1_weighted
        ))

    # EDA BackTranslation GPT3Mix AugGPT LLM2LLM ours_l_r ours_r_l ours_r
    print("\n\nCaculate semantic fidelity")
    for augmenter in ["EDA", "BackTranslation", "GPT3Mix", "AugGPT", "LLM2LLM", "ours_l_r", "ours_r_l", "ours_r"]:
        print(f"########## {augmenter}")

        accuracy_result = []
        f1_macro_result = []

        for exp_id in range(10):

            current_file_path = __file__
            current_dir = os.path.dirname(current_file_path)
            task_name = "classification"
            seednum = int(args.seednum)

            base_path = f'{current_dir}/../../datasets/{task_name}/{args.dataset}/'

            test_path = base_path + f'data_augmentation/augmentation_num_{args.aug_num:02}/exp_{exp_id:02}/{augmenter}/' + \
                         f"train_subsample_{seednum:03}_{augmenter}.jsonl"
            examples_test = create_examples(read_jsonl(test_path), "test")


            classifier._dataset = {}
            classifier._data_loader = {}
            classifier.load_data('test', examples_test, args.batch_size, max_length=args.max_seq_length, shuffle=False)

            results = classifier.evaluate('test')
            final_test_acc = results["acc"]

            accuracy = results["accuracy"]
            f1_micro = results["f1_micro"]
            f1_macro = results["f1_macro"]
            f1_weighted = results["f1_weighted"]
            print('Test Acc: {:.4f}  accuracy: {:.4f}  f1_micro: {:.4f}  f1_macro: {:.4f}  f1_weighted: {:.4f}'.format(
                    100. * final_test_acc, 
                    100. * accuracy, 
                    100. * f1_micro,
                    100. * f1_macro, 
                    100. * f1_weighted))

            accuracy_result.append(100. * accuracy)
            f1_macro_result.append(100. * f1_macro)    

        accuracy_mean = sum(accuracy_result) / len(accuracy_result)
        accuracy_variance = sum((x - accuracy_mean) ** 2 for x in accuracy_result) / len(accuracy_result)
        accuracy_std_deviation = accuracy_variance ** 0.5

        print(f"semantic fidelity: {accuracy_result}")
        print(f"Mean semantic fidelity: {accuracy_mean:.2f}({accuracy_std_deviation:.2f})")
        # print(f"Mean semantic fidelity: {100*accuracy_mean:.2f}({accuracy_std_deviation:.2f})")

        # f1_macro_mean = sum(f1_macro_result) / len(f1_macro_result)
        # f1_macro_variance = sum((x - f1_macro_mean) ** 2 for x in f1_macro_result) / len(f1_macro_result)
        # f1_macro_std_deviation = f1_macro_variance ** 0.5
        # print(f"Mean f1_macro: {f1_macro_mean:.2f}({f1_macro_std_deviation:.2f})")


import json
def read_jsonl(input_file):
    if input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

    return data


def create_examples(data, set_type):

    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(data):
        guid = "%s-%s" % (set_type, i)
        text_a = line["text"]
        label = line["label"]
        examples.append(
            InputExample(guid=guid, text_a=text_a, label=label))
    return examples



if __name__ == '__main__':
    """
    # Baseline classifier
    python semantic_fidelity.py --cuda 0 --dataset SNIPS --exp_id 0 --augmenter none --seednum 70 --cache cache > bert_baseline.log
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', default=1, type=int)
    parser.add_argument('--dataset')#, choices=['SST2', 'SNIPS', 'TREC'])


    parser.add_argument("--exp_id", default=1, type=int)
    parser.add_argument("--aug_num", default=3, type=int)

    parser.add_argument("--aug_num_sub", default=None, type=int)

    parser.add_argument("--augmenter", default="none", type=str)
    parser.add_argument("--seednum", default=50, type=float)

    parser.add_argument('--cache', default="transformers_cache", type=str)


    parser.add_argument('--hidden_dropout_prob', default=0.1, type=float)
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--max_seq_length", default=64, type=int,
                        help="The maximum total input sequence length after tokenization. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")


    parser.add_argument('--epochs', default=8, type=int)
    parser.add_argument('--min_epochs', default=0, type=int)
    parser.add_argument("--learning_rate", default=4e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)

    parser.add_argument("--model_type", default="bert-base-uncased", type=str)

    args = parser.parse_args()

    if args.aug_num_sub == None:
        args.aug_num_sub = args.aug_num

    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if cuda else 'cpu')
    print('device:', device)

    main(args, device)

