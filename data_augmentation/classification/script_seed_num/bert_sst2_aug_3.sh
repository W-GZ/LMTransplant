#!/usr/bin/env bash

# cd data_augmentation/classification/
# bash script_seed_num/bert_sst2_aug_3.sh

SRC=bert_model
CACHE=cache

augmentator_list="none MoreData EDA BackTranslation GPT3Mix AugGPT LLM2LLM ours_l_r ours_r_l ours_r"


DATASET=SST2
SEEDNUMS="40 60 80 100"

for augmenter in $augmentator_list; do
    for SEEDNUM in $SEEDNUMS; do

    RAWDATADIR=../datasets/classification/${DATASET}/data_augmentation/augmentation_num_03/exp_0${i}/${augmenter}

    # ModernBERT-base
    python $SRC/bert_classifier.py --cuda 2 --dataset $DATASET --exp_id ${i} --augmenter $augmenter --seednum ${SEEDNUM} --cache $CACHE > $RAWDATADIR/train_subsample_0${SEEDNUM}_${augmenter}_ModernBERT-base.log --model_type ModernBERT-base

    done
done


