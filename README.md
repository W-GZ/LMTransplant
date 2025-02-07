# Left, Right, and Tens of Thousands: A New Paradigm for Text Data Augmentation
This is the PyTorch implementation for the following paper: **Left, Right, and Tens of Thousands: A New Paradigm for Text Data Augmentation**.


## Introduction
This repo provides the code for reproducing the experiments in “Left, Right, and Tens of Thousands: A New Paradigm for Text Data Augmentation”. 
LMTransplant is a novel Data Augmentation method with LLM-based text transplanting. 
It crafts realistic contextual scenarios to the original text, by leveraging external knowledge in LLMs, thereby crafting higher-quality and more diverse text data.


## Overview
We develop LMTransplant, a novel text data augmentation paradigm based on transplantation. Following illustrates the overall pipeline.
LMTransplant generates high-quality and diverse augmented text by leveraging a bidirectional text completion strategy and masked text prediction. 
It generates contextually relevant scenes that align with the original text, making full use of the external knowledge embedded in LLMs. 
We elaborate on each step in the following sections.

<img src="overview.png" width="750"><br/>


## Datasets download and preprocessing

```bash
git clone https://anonymous.4open.science/r/
cd LMTransplant
pip install -r requirements.txt

cd data_augmentation/classification/utils
bash download_and_prepare_datasets.sh

cd data_augmentation
python datasets_preprocess.py
```


## Get seed data

```bash
cd data_augmentation
python generate_seed_data.py
```


## Generate augmented data

```bash
cd data_augmentation
python None_MoreData.py
python EDA.py
python ours_l_r.py
```


## Intrinsic evaluation

#### Distinct-n
```bash
cd data_augmentation/eval_distinct_n
python distinct_3.py
```

#### Semantic fidelity
```bash
cd data_augmentation/classification
bash script_semantic_fidelity/semantic_fidelity_ModernBERT_base.sh
```


## Extrinsic evaluation

#### Classification task
```bash
cd data_augmentation/classification
bash script/bert_sst2.sh
```

#### Question answering task
```bash
cd data_augmentation/question_answer
bash 
```



## Main Result
<img src="result.png" width="750"><br/>


