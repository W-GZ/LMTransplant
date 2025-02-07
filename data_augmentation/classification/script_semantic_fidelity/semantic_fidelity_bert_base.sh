#!/usr/bin/env bash

# cd data_augmentation/classification/
# bash script_semantic_fidelity/semantic_fidelity_bert_base.sh

SRC=bert_model
CACHE=cache


DATASET=SNIPS
SEEDNUM=70
python $SRC/semantic_fidelity.py --cuda 0 --dataset $DATASET --exp_id 0 --seednum ${SEEDNUM} --cache $CACHE > snips_semantic_fidelity_bert-base-uncased.log --model_type bert-base-uncased


DATASET=SST2
SEEDNUM=20
python $SRC/semantic_fidelity.py --cuda 0 --dataset $DATASET --exp_id 0 --seednum ${SEEDNUM} --cache $CACHE > sst2_semantic_fidelity_bert-base-uncased.log --model_type bert-base-uncased


DATASET=TREC
SEEDNUM=60
python $SRC/semantic_fidelity.py --cuda 0 --dataset $DATASET --exp_id 0 --seednum ${SEEDNUM} --cache $CACHE > trec_semantic_fidelity_bert-base-uncased.log --model_type bert-base-uncased



