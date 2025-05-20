
augmentator_list="none MoreData EDA"

DATASET=CONLL2003
SEEDNUM=50

for augmenter in $augmentator_list; do
    for i in {0..9}; do

    RAWDATADIR=../datasets/ner/${DATASET}/data_augmentation/augmentation_num_03/exp_0${i}/${augmenter}

    # ModernBERT-base
    python train.py --cuda 1 --dataset $DATASET --aug_num 3 --exp_id ${i} --augmenter $augmenter --seednum ${SEEDNUM} --model_type ModernBERT-base > $RAWDATADIR/train_subsample_0${SEEDNUM}_${augmenter}_ModernBERT-base.log

    done
done
