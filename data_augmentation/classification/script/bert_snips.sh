#!/usr/bin/env bash

# cd data_augmentation/classification/
# bash script/bert_snips.sh

SRC=bert_model
CACHE=cache
augmentator_list="none MoreData EDA BackTranslation GPT3Mix AugGPT LLM2LLM ours_l_r ours_r_l ours_r"

DATASET=SNIPS
SEEDNUM=70

for augmenter in $augmentator_list; do
    for i in {0..9}; do

    RAWDATADIR=../datasets/classification/${DATASET}/data_augmentation/augmentation_num_03/exp_0${i}/${augmenter}

    # BERT-base-uncased
    # python $SRC/bert_classifier.py --cuda 0 --dataset $DATASET --exp_id ${i} --augmenter $augmenter --seednum ${SEEDNUM} --cache $CACHE > $RAWDATADIR/train_subsample_0${SEEDNUM}_${augmenter}_BERT-base-uncased.log

    # ModernBERT-base
    python $SRC/bert_classifier.py --cuda 0 --dataset $DATASET --exp_id ${i} --augmenter $augmenter --seednum ${SEEDNUM} --cache $CACHE > $RAWDATADIR/train_subsample_0${SEEDNUM}_${augmenter}_ModernBERT-base.log --model_type ModernBERT-base

    # ModernBERT-large
    # python $SRC/bert_classifier.py --cuda 0 --dataset $DATASET --exp_id ${i} --augmenter $augmenter --seednum ${SEEDNUM} --cache $CACHE > $RAWDATADIR/train_subsample_0${SEEDNUM}_${augmenter}_ModernBERT-large.log --model_type ModernBERT-large

    done
done


