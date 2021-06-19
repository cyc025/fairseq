#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


SRC=data
TGT=en
CORPUS_PATH=~/fairseq_test_data

# create result dir
mkdir -p results
mkdir -p $CORPUS_PATH/st_corpus

CORPUS_DEST=${SRC}_${TGT}

TRAIN_SRC_PATH=$CORPUS_PATH/train.$SRC
TEST_SRC_PATH=$CORPUS_PATH/test.$SRC
DEV_SRC_PATH=$CORPUS_PATH/valid.$SRC

TRAIN_TGT_PATH=$CORPUS_PATH/train.$TGT
TEST_TGT_PATH=$CORPUS_PATH/test.$TGT
DEV_TGT_PATH=$CORPUS_PATH/valid.$TGT

# define preprocessed corpus
TRAINSET_PATH=$CORPUS_PATH/$CORPUS_DEST

# combine src and tgt
cat $TRAIN_SRC_PATH $TRAIN_TGT_PATH $DEV_SRC_PATH $DEV_TGT_PATH $TEST_SRC_PATH $TEST_TGT_PATH > $CORPUS_PATH/.$SRC-$TGT

SPM_MODEL_PATH=$CORPUS_PATH/$SRC-$TGT;
SPM_MODEL=$SPM_MODEL_PATH.model;

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
rm -f $CORPUS_PATH/pretrain/*
