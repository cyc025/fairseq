#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Under CORPUS_PATH, create st_corpus to store monolingual source data.
# Name all files as [train|dev|test].[$SRC|$TGT]
#


set -e



SRC=en_XX
TGT=de_DE
CORPUS_PATH=~/de

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
rm -f results/*
rm -f $CORPUS_PATH/pretrain/*

TRAIN_SRC_PATH=$CORPUS_PATH/train.$SRC
TEST_SRC_PATH=$CORPUS_PATH/test.$SRC
DEV_SRC_PATH=$CORPUS_PATH/dev.$SRC

TRAIN_TGT_PATH=$CORPUS_PATH/train.$TGT
TEST_TGT_PATH=$CORPUS_PATH/test.$TGT
DEV_TGT_PATH=$CORPUS_PATH/dev.$TGT

# define preprocessed corpus
TRAINSET_PATH=$CORPUS_PATH/$CORPUS_DEST


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
mv $TRAIN_SRC_PATH.spm $CORPUS_PATH/train.spm.$SRC
mv $TRAIN_TGT_PATH.spm $CORPUS_PATH/train.spm.$TGT
mv $DEV_SRC_PATH.spm $CORPUS_PATH/dev.spm.$SRC
mv $DEV_TGT_PATH.spm $CORPUS_PATH/dev.spm.$TGT
mv $TEST_SRC_PATH.spm $CORPUS_PATH/test.spm.$SRC
mv $TEST_TGT_PATH.spm $CORPUS_PATH/test.spm.$TGT


DICT=mbart/cc25_pretrain/dict.txt
preprocess() {
    python preprocess.py \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref $CORPUS_PATH/train.spm \
    --validpref $CORPUS_PATH/dev.spm \
    --testpref $CORPUS_PATH/test.spm  \
    --destdir $CORPUS_PATH/$CORPUS_DEST \
    --srcdict ${DICT} \
    --tgtdict ${DICT} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --workers 70;
}

preprocess;
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
model=mbart/cc25_pretrain/model.pt
python generate.py $TRAINSET_PATH \
--path $model  --task translation_from_pretrained_bart \
--gen-subset test -t $TGT -s $SRC --bpe 'sentencepiece' \
--sentencepiece-vocab $SPM_MODEL --sacrebleu  \
--remove-bpe 'sentencepiece' --max-sentences 32 \
--langs $langs
