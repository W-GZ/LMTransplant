#!/usr/bin/env bash

# cd data_augmentation/classification/
# bash script_semantic_fidelity/semantic_fidelity_ModernBERT_large.sh

SRC=bert_model
CACHE=cache


DATASET=SNIPS
SEEDNUM=70
python $SRC/semantic_fidelity.py --cuda 2 --dataset $DATASET --exp_id 0 --seednum ${SEEDNUM} --cache $CACHE > snips_semantic_fidelity_ModernBERT-large.log --model_type ModernBERT-large


DATASET=SST2
SEEDNUM=20
python $SRC/semantic_fidelity.py --cuda 2 --dataset $DATASET --exp_id 0 --seednum ${SEEDNUM} --cache $CACHE > sst2_semantic_fidelity_ModernBERT-large.log --model_type ModernBERT-large


DATASET=TREC
SEEDNUM=60
python $SRC/semantic_fidelity.py --cuda 2 --dataset $DATASET --exp_id 0 --seednum ${SEEDNUM} --cache $CACHE > trec_semantic_fidelity_ModernBERT-large.log --model_type ModernBERT-large

