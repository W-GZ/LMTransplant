#!/usr/bin/env bash

# cd data_augmentation/classification/
# bash script_augmented_num/bert_sst2_aug_10.sh

SRC=bert_model
CACHE=cache

augmentator_list="none MoreData EDA BackTranslation GPT3Mix AugGPT LLM2LLM ours_l_r ours_r_l ours_r"


DATASET=SST2
SEEDNUM=20

for augmenter in $augmentator_list; do
    for i in {0..10}; do

    RAWDATADIR=../datasets/classification/${DATASET}/data_augmentation/augmentation_num_10/exp_0${i}/${augmenter}

    # ModernBERT-base
    python $SRC/bert_classifier.py --cuda 2 --dataset $DATASET --exp_id ${i} --aug_num 10 --aug_num_sub ${i} --augmenter $augmenter --seednum ${SEEDNUM} --cache $CACHE > $RAWDATADIR/${i}.log --model_type ModernBERT-base

    done
done


