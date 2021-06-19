#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Under CORPUS_PATH, create st_corpus to store monolingual source data.
# Name all files as [train|dev|test].[$SRC|$TGT]
#
#export CUDA_VISIBLE_DEVICES="0"
set -e

SRC=$1
TGT=$2
#CORPUS_PATH=~/wikibio-xx-master/test_fr
CORPUS_PATH=~/wikibio-xx-master/$TGT

# create result dir
mkdir -p results
mkdir -p $CORPUS_PATH/st_corpus

CORPUS_DEST=${SRC}_${TGT}

# clean files
rm -f $CORPUS_PATH/$CORPUS_DEST/*
rm -f $CORPUS_PATH/preprocess.log
rm -f $CORPUS_PATH/*spm*
rm -f $CORPUS_PATH/*.vocab
rm -f $CORPUS_PATH/*.model
rm -f $CORPUS_PATH/*.bin
rm -f $CORPUS_PATH/*.idx
rm -f $CORPUS_PATH/*.txt
rm -f $CORPUS_PATH/st_corpus/*spm*
rm -f $CORPUS_PATH/st_corpus/*.txt
rm -f $CORPUS_PATH/st_corpus/*.bin
rm -f $CORPUS_PATH/st_corpus/*.idx
rm -f results/*

TRAIN_SRC_PATH=$CORPUS_PATH/new_train.$SRC.tok.split
TEST_SRC_PATH=$CORPUS_PATH/new_test.$SRC.tok.split
DEV_SRC_PATH=$CORPUS_PATH/new_valid.$SRC.tok.split

TRAIN_TGT_PATH=$CORPUS_PATH/new_train.$TGT.tok.split
TEST_TGT_PATH=$CORPUS_PATH/new_test.$TGT.tok.split
DEV_TGT_PATH=$CORPUS_PATH/new_valid.$TGT.tok.split

# define preprocessed corpus
TRAINSET_PATH=$CORPUS_PATH/$CORPUS_DEST

# combine src and tgt
cat $TRAIN_SRC_PATH $TRAIN_TGT_PATH  > $CORPUS_PATH/.$SRC-$TGT


build_spm_model() {
    SPM_MODEL_PATH=$CORPUS_PATH/$SRC-$TGT
    SPM_MODEL=$SPM_MODEL_PATH.model
    echo "Building sentencepiece model.";
    python sentence_piece.py --mode build \
    --model_path $SPM_MODEL_PATH \
    --corpus $CORPUS_PATH/.$SRC-$TGT
    # define model path and remove created corpus
    rm $CORPUS_PATH/.$SRC-$TGT;
}
SPM_MODEL_PATH=mbart/cc25_pretrain/sentence.bpe
SPM_MODEL=mbart/cc25_pretrain/sentence.bpe.model

echo "Converting source documents to sentencepieces."
python sentence_piece.py --mode doc2spm \
--model_path $SPM_MODEL_PATH \
--corpus $TRAIN_SRC_PATH
python sentence_piece.py --mode doc2spm \
--model_path $SPM_MODEL_PATH \
--corpus $TRAIN_TGT_PATH
python sentence_piece.py --mode doc2spm \
--model_path $SPM_MODEL_PATH \
--corpus $DEV_SRC_PATH
python sentence_piece.py --mode doc2spm \
--model_path $SPM_MODEL_PATH \
--corpus $DEV_TGT_PATH
python sentence_piece.py --mode doc2spm \
--model_path $SPM_MODEL_PATH \
--corpus $TEST_SRC_PATH
python sentence_piece.py --mode doc2spm \
--model_path $SPM_MODEL_PATH \
--corpus $TEST_TGT_PATH

# rename paths for preprocessing
mv $TRAIN_SRC_PATH.spm $CORPUS_PATH/new_train.spm.$SRC
mv $TRAIN_TGT_PATH.spm $CORPUS_PATH/new_train.spm.$TGT
mv $DEV_SRC_PATH.spm $CORPUS_PATH/new_valid.spm.$SRC
mv $DEV_TGT_PATH.spm $CORPUS_PATH/new_valid.spm.$TGT
mv $TEST_SRC_PATH.spm $CORPUS_PATH/new_test.spm.$SRC
mv $TEST_TGT_PATH.spm $CORPUS_PATH/new_test.spm.$TGT

preprocess() {
    python preprocess.py \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref $CORPUS_PATH/new_train.spm \
    --validpref $CORPUS_PATH/new_valid.spm \
    --testpref $CORPUS_PATH/new_test.spm  \
    --destdir $CORPUS_PATH/$CORPUS_DEST \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --tgtdict ~/wikibio-xx-master/fr/en_fr/dict.fr.txt \
    --srcdict ~/wikibio-xx-master/fr/en_fr/dict.en.txt \
    --workers 70;
}

# Model configurations
WARMUP_UPDATES=50;
PATIENCE=20;
TOTAL_EPOCH=200;

TASK=translation;
ARCH=transformer_wmt_en_de;


pretrain() {
    python train.py $1  \
    --source-lang $2 --target-lang $3 \
    --task $4 \
    --arch $ARCH \
    --fp16 \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates $WARMUP_UPDATES \
    --patience=$PATIENCE \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2048 \
    --skip-invalid-size-inputs-valid-test \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --log-interval 2 \
    --max-epoch $TOTAL_EPOCH \
    --save-dir checkpoints/$SRC-$TGT \
    --no-epoch-checkpoints \
    --no-last-checkpoints;
}

train() {
    python train.py $1  \
    --source-lang $2 --target-lang $3 \
    --task $4 \
    --arch $ARCH \
    --fp16 \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates $WARMUP_UPDATES \
    --patience=$PATIENCE \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2048 \
    --skip-invalid-size-inputs-valid-test \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric \
    --log-interval 2 \
    --restore-file checkpoints/$SRC-$TGT/checkpoint_best.pt \
    --max-epoch 300 \
    --save-dir checkpoints/$SRC-$TGT \
    --no-epoch-checkpoints \
    --no-last-checkpoints;
}

mkdir -p outputs/$SRC-$TGT
model=checkpoints/$SRC-$TGT/checkpoint_best.pt
generate() {
    python generate.py $1 \
    --path $model --task $TASK \
    --gen-subset test -t $TGT -s $SRC --bpe 'sentencepiece' \
    --sentencepiece-vocab $SPM_MODEL \
    --max-sentences 32 \
     --num-workers 70 \
     --beam 5 \
     --results-path outputs/$2 \
     --skip-invalid-size-inputs-valid-test \
     --source-lang $SRC --target-lang $TGT;
}


####################
## Self-Training  ##
####################
COMB_DATA=data-bin/para_plus_st
CURR_TRAINSET_PATH=$TRAINSET_PATH
echo "Preprocessing documents."
preprocess;
echo "Start self-training loop."
for I in 1 2 3 4 5 6 7 8 9 10
do
    echo "Pre-training model."
    pretrain $CURR_TRAINSET_PATH $SRC $TGT $TASK;
    echo "Training model."
    train $TRAINSET_PATH $SRC $TGT $TASK;
    echo "Generate for step I."
    generate $TRAINSET_PATH $I
    echo "Clean previous environments."
    rm -rf $COMB_DATA/*
    echo "Augmentation step."
    bash run_st_aug.sh $SRC $TGT ~/wikibio-xx-master/fr/all.en.tok.split
    echo "Changing train set path to...${COMB_DATA}";
    CURR_TRAINSET_PATH=$COMB_DATA/$CORPUS_DEST
    echo "${CURR_TRAINSET_PATH}"
    rm checkpoints/$SRC-$TGT/*
done
