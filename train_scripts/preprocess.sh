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
CORPUS_PATH=$3

# create result dir
mkdir -p results

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

# define self-training source, to be modified

TRAIN_SRC_PATH=$CORPUS_PATH/new_train.$SRC.tok.split
TEST_SRC_PATH=$CORPUS_PATH/new_test.$SRC.tok.split
DEV_SRC_PATH=$CORPUS_PATH/new_valid.$SRC.tok.split

TRAIN_TGT_PATH=$CORPUS_PATH/new_train.$TGT.tok.split
TEST_TGT_PATH=$CORPUS_PATH/new_test.$TGT.tok.split
DEV_TGT_PATH=$CORPUS_PATH/new_valid.$TGT.tok.split

# define preprocessed corpus
TRAINSET_PATH=$CORPUS_PATH/$CORPUS_DEST

# combine src and tgt
cat $TRAIN_SRC_PATH $TRAIN_TGT_PATH > $CORPUS_PATH/.$SRC-$TGT


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
SPM_MODEL_PATH=~/fairseq/mbart/cc25_pretrain/sentence.bpe
SPM_MODEL=~/fairseq/mbart/cc25_pretrain/sentence.bpe.model

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
    --joined-dictionary \
    --workers 70;
}

preprocess;
