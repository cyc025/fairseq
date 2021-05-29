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
CORPUS_PATH=~/wikibio-xx-master/$TGT

# create result dir
mkdir -p results

CORPUS_DEST=${SRC}_${TGT}

TRAIN_SRC_PATH=$CORPUS_PATH/new_train.$SRC
TEST_SRC_PATH=$CORPUS_PATH/new_test.$SRC
DEV_SRC_PATH=$CORPUS_PATH/new_valid.$SRC

TRAIN_TGT_PATH=$CORPUS_PATH/new_train.$TGT.tok.split
TEST_TGT_PATH=$CORPUS_PATH/new_test.$TGT.tok.split
DEV_TGT_PATH=$CORPUS_PATH/new_valid.$TGT.tok.split

# define preprocessed corpus
TRAINSET_PATH=$CORPUS_PATH/$CORPUS_DEST


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


# Model configurations
WARMUP_UPDATES=50;
PATIENCE=100;
TOTAL_EPOCH=1000;

TASK=translation;
ARCH=transformer_wmt_en_de;

model=checkpoints/$SRC-$TGT/checkpoint_best.pt


train_translate() {
    python train.py $1  \
    --source-lang $2 --target-lang $3 \
    --task $4 \
    --arch $ARCH \
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
    --restore-file $model \
    --no-epoch-checkpoints \
    --no-last-checkpoints;
}


####################
## Self-Training  ##
####################
CURR_TRAINSET_PATH=$TRAINSET_PATH
echo "Training model."
train_translate $CURR_TRAINSET_PATH $SRC $TGT $TASK;
