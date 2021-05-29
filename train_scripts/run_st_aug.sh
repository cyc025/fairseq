#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Under CORPUS_PATH, create st_corpus to store monolingual source data.
# Name all files as [train|dev|test].[$SRC|$TGT]
#

set -e

SRC=$1
TGT=$2
MONO_SRC=$3
CORPUS_PATH=~/wikibio-xx-master/$TGT

# create result dir
mkdir -p results
mkdir -p $CORPUS_PATH/st_corpus

CORPUS_DEST=${SRC}_${TGT}

# clean files
rm -f $CORPUS_PATH/st_corpus/*spm*
rm -f $CORPUS_PATH/st_corpus/*.txt
rm -f $CORPUS_PATH/st_corpus/*.bin
rm -f $CORPUS_PATH/st_corpus/*.idx


# define self-training source, to be modified
ST_TRAIN_SRC_PATH=$CORPUS_PATH/st_corpus/mono.$SRC
cp $MONO_SRC $ST_TRAIN_SRC_PATH

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

SPM_MODEL_PATH=mbart/cc25_pretrain/sentence.bpe
SPM_MODEL=mbart/cc25_pretrain/sentence.bpe.model

echo "Converting source documents to sentencepieces."
python sentence_piece.py --mode doc2spm \
--model_path $SPM_MODEL_PATH \
--corpus $ST_TRAIN_SRC_PATH

preprocess_self_training() {
    python sentence_piece.py --mode doc2spm \
    --model_path $SPM_MODEL_PATH \
    --corpus $CORPUS_PATH/st_corpus/mono.$SRC;
    mv $CORPUS_PATH/st_corpus/mono.$SRC.spm $CORPUS_PATH/st_corpus/mono.spm.$SRC;
    python preprocess.py \
    --only-source \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --srcdict $CORPUS_PATH/$CORPUS_DEST/dict.$SRC.txt \
    --testpref $CORPUS_PATH/st_corpus/mono.spm \
    --destdir $CORPUS_PATH/st_corpus \
    --thresholdsrc 0 \
    --joined-dictionary \
    --workers 70;
}

# Model configurations
WARMUP_UPDATES=1000;
TOTAL_NUM_UPDATE=2000;
TASK=translation;
ARCH=transformer_wmt_en_de;

model=checkpoints/$SRC-$TGT/checkpoint_best.pt

generate_for_self_training() {
     cp $CORPUS_PATH/$CORPUS_DEST/dict.$TGT.txt $CORPUS_PATH/st_corpus/.
     python generate.py $CORPUS_PATH/st_corpus \
     --path $model --task $TASK \
     --gen-subset test -t $TGT -s $SRC --bpe 'sentencepiece' \
     --sentencepiece-vocab $SPM_MODEL \
     --max-tokens 2048 \
     --max-sentences 512 \
      --num-workers 70 \
      --sampling \
      --sampling-topk 5 \
      --beam 5 \
      --nbest 5 \
      --num-shards 8 --shard-id 7 \
      --model-parallel-size 8 \
      --skip-invalid-size-inputs-valid-test \
       --results-path $CORPUS_PATH/st_corpus \
      --source-lang $SRC --target-lang $TGT;
}

## Prepare Self-training
ST_OUTPUTS=self_train_outputs
ST_CORPUS=data-bin/$ST_OUTPUTS
# create result dir
mkdir -p $ST_CORPUS

# Create directories for augmentation
PARA_DATA=$(readlink -f $CORPUS_PATH/$CORPUS_DEST)
ST_DATA=$(readlink -f $ST_CORPUS)
COMB_DATA=data-bin/para_plus_st
mkdir -p $COMB_DATA

# 1. convert source file to .spm
# 2. preprocess both source and target spm files
self_train_preprocess() {

    cat $ST_CORPUS/$SRC $TRAIN_SRC_PATH > $COMB_DATA/$SRC
    cat $ST_CORPUS/$TGT $TRAIN_TGT_PATH > $COMB_DATA/$TGT

    python sentence_piece.py --mode doc2spm \
    --model_path $SPM_MODEL_PATH \
    --corpus $COMB_DATA/$SRC;
    python sentence_piece.py --mode doc2spm \
    --model_path $SPM_MODEL_PATH \
    --corpus $COMB_DATA/$TGT;

    mv $COMB_DATA/$SRC.spm $COMB_DATA/new_train.spm.$SRC;
    mv $COMB_DATA/$TGT.spm $COMB_DATA/new_train.spm.$TGT;

    python preprocess.py \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref $COMB_DATA/new_train.spm \
    --validpref $CORPUS_PATH/new_valid.spm \
    --testpref $CORPUS_PATH/new_test.spm  \
    --destdir ${COMB_DATA}/$CORPUS_DEST \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --tgtdict $COMB_DATA/dict.fr.txt \
    --srcdict $COMB_DATA/dict.en.txt \
    --workers 70;
}

########################
## Start of pipeline  ##
########################

echo "Preprocessing documents."
preprocess_self_training;

####################
## Self-Training  ##
####################
CURR_TRAINSET_PATH=$TRAINSET_PATH
echo "Start self-training loop."

# clean previous augmentations
#rm -f $ST_CORPUS/*
#rm -f $COMB_DATA/*
echo "Generating ST outputs."
generate_for_self_training;
echo "Creating self-training parallel data."
grep 'H-' $CORPUS_PATH/st_corpus/generate-test.txt | awk -F '\t' '{print $3}' > $ST_CORPUS/$TGT
grep 'S-' $CORPUS_PATH/st_corpus/generate-test.txt | awk -F '\t' '{print $2}' > $ST_CORPUS/$SRC
echo "Preprocessing parallel data.";
# copy dictionaries
cp ~/wikibio-xx-master/fr/en_fr/dict.fr.txt $COMB_DATA/dict.fr.txt
cp ~/wikibio-xx-master/fr/en_fr/dict.en.txt $COMB_DATA/dict.en.txt
self_train_preprocess;
