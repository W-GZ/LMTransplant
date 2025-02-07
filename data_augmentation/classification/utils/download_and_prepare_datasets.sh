#!/usr/bin/env bash

# cd classification/utils
# bash download_and_prepare_datasets.sh

mkdir datasets


# TREC dataset
mkdir -p datasets/TREC
for split in train dev test;
  do
    wget -O datasets/TREC/${split}.raw  https://raw.githubusercontent.com/1024er/cbert_aug/crayon/datasets/TREC/${split}.tsv
    python convert_num_to_text_labels.py -i datasets/TREC/${split}.raw -o datasets/TREC/${split}.tsv -d TREC
    rm datasets/TREC/${split}.raw
  done


# SST2 dataset
mkdir -p datasets/SST2
for split in train dev test;
  do
    wget -O datasets/SST2/${split}.raw  https://raw.githubusercontent.com/1024er/cbert_aug/crayon/datasets/stsa.binary/${split}.tsv
    python convert_num_to_text_labels.py -i datasets/SST2/${split}.raw -o datasets/SST2/${split}.tsv -d SST2
    rm datasets/SST2/${split}.raw
  done


# SNIPS dataset
mkdir -p datasets/SNIPS
for split in train valid test;
  do
    wget -O datasets/SNIPS/${split}.seq  https://raw.githubusercontent.com/MiuLab/SlotGated-SLU/master/data/snips/${split}/seq.in
    wget -O datasets/SNIPS/${split}.label  https://raw.githubusercontent.com/MiuLab/SlotGated-SLU/master/data/snips/${split}/label
    paste -d'\t' datasets/SNIPS/${split}.label datasets/SNIPS/${split}.seq  > datasets/SNIPS/${split}.tsv
    rm datasets/SNIPS/${split}.label
    rm datasets/SNIPS/${split}.seq
  done

mv datasets/SNIPS/valid.tsv datasets/SNIPS/dev.tsv

