# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0


import torch, os
import argparse

from data_processors import get_data
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

if __name__ == '__main__':
    """
    # Baseline classifier
    python bert_classifier.py --cuda 0 --dataset SNIPS --exp_id 0 --augmenter none --seednum 70 --cache cache > bert_baseline.log
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

