# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import random
import os, json
import csv


current_file_path = __file__
current_dir = os.path.dirname(current_file_path)


def get_task_processor(args):
    """
    A TSV processor for stsa, trec and snips dataset.
    """
    task = args.dataset
    if task == 'SST2':
        return TSVDataProcessor(args=args, skip_header=False, label_col="label", text_col="text")
    elif task == 'TREC':
        return TSVDataProcessor(args=args, skip_header=False, label_col="label", text_col="text")
    elif task == 'SNIPS':
        return TSVDataProcessor(args=args, skip_header=False, label_col="label", text_col="text")
    else:
        # raise ValueError('Unknown task')
        return TSVDataProcessor(args=args, skip_header=False, label_col="label", text_col="text")


def get_data(args):
    random.seed(args.exp_id)
    processor = get_task_processor(args)

    examples = dict()

    examples['train'] = processor.get_train_examples()
    examples['dev'] = processor.get_dev_examples()
    examples['test'] = processor.get_test_examples()

    for key, value in examples.items():
        print('#{}: {}'.format(key, len(value)))
    return examples, processor.get_labels()


class InputExample:
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

    def __getitem__(self, item):
        return [self.input_ids, self.input_mask,
                self.segment_ids, self.label_id][item]


class DatasetProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, task_name):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
    
    @classmethod
    def _read_jsonl(cls, input_file):
        if input_file.endswith('.json'):
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]

        return data



class TSVDataProcessor(DatasetProcessor):
    """Processor for dataset to be augmented."""

    def __init__(self, args, skip_header, label_col, text_col):
        self.args = args

        task_name="classification"
        seednum = int(args.seednum)

        base_path = f'{current_dir}/../../datasets/{task_name}/{args.dataset}/'

        self.train_path = base_path + f'data_jsonl/train.jsonl'
        self.dev_path = base_path + f'data_jsonl/dev.jsonl'
        self.test_path = base_path + f'data_jsonl/test.jsonl'

        self.skip_header = skip_header
        self.label_col = label_col
        self.text_col = text_col

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(self._read_jsonl(self.train_path), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(self._read_jsonl(self.dev_path), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(self._read_jsonl(self.test_path), "test")

    def get_labels(self):
        """add your dataset here"""
        # labels_dict = {
        #     'SNIPS': ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent'],
        #     'SST2': ['Negative', 'Positive'],
        #     'TREC': ['Abbreviation', 'Description', 'Entity', 'Human', 'Location', 'Numeric'],
        # }

        # labels = labels_dict[self.args.dataset]
        # return sorted(labels)


        labels = set()
        train_data = self._read_jsonl(self.train_path)
        for item in train_data:
            labels.add(item[self.label_col])

        print(labels)
        print(sorted(labels))
        return sorted(labels)


    def _create_examples(self, lines, set_type):
        if set_type == "train":
            aug_num_sub = self.args.aug_num_sub
            aug_num = self.args.aug_num

            if aug_num_sub == aug_num:
                data = lines
            else:
                if self.args.augmenter == "MoreData":
                    data = lines[:20*(aug_num_sub + 1)]
                else:

                    data_all = []
                    data_temp = []
                    
                    for item in lines:
                        if "aug_sample_flag" not in item:
                            if data_temp == []:
                                data_temp.append(item)
                            else:
                                data_all.append(data_temp)
                                data_temp = []
                                data_temp.append(item)
                        else:
                            data_temp.append(item)
                    
                    data_all.append(data_temp)

                    data = []
                    for item in data_all:
                        data.extend(item[:(aug_num_sub + 1)])
        else:
            data = lines

        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(data):
            if self.skip_header and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[self.text_col]
            label = line[self.label_col]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
        return examples

