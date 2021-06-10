#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Under CORPUS_PATH, create st_corpus to store monolingual source data.
# Name all files as [train|dev|test].[$SRC|$TGT]
#


set -e

# kill all current processes
# nvidia-smi | grep 'python' | awk '{ print $3 }' | xargs -n1 kill -9


SRC=src
TGT=tgt
CORPUS_PATH=~/cycle_data

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


# clean
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


SPM_MODEL_PATH=$CORPUS_PATH/$SRC-$TGT;
SPM_MODEL=$SPM_MODEL_PATH.model;
build_spm_model() {
    echo "Building sentencepiece model.";
    python sentence_piece.py --mode build \
    --model_path $SPM_MODEL_PATH \
    --corpus $CORPUS_PATH/.$SRC-$TGT;
    # define model path and remove created corpus
    rm $CORPUS_PATH/.$SRC-$TGT;
}

docConverter() {
    build_spm_model;
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
}

preprocess() {
    docConverter;
    fairseq-preprocess \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref $CORPUS_PATH/train.spm \
    --validpref $CORPUS_PATH/dev.spm \
    --testpref $CORPUS_PATH/test.spm  \
    --destdir $CORPUS_PATH/$CORPUS_DEST \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --joined-dictionary \
    --workers 70;
}


# Model configurations
WARMUP_UPDATES=50;
TOTAL_NUM_UPDATE=100;
PATIENCE=100;
TOTAL_EPOCH=300;
TASK=translation_lev;
ARCH=nonautoregressive_transformer;
LR=0.0005

train_translate() {
    python3 train.py $1 \
    --batch-size 128 \
    --save-dir checkpoints \
    --ddp-backend=no_c10d \
    --task $TASK \
    --criterion nat_loss \
    --arch $ARCH \
    --noise random_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr $LR --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'tqdm' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 8000 \
    --no-epoch-checkpoints \
    --save-interval-updates 10000 \
    --max-update 10000;
}


model=checkpoints/checkpoint_best.pt
generate() {
    fairseq-generate $CORPUS_PATH/$CORPUS_DEST \
    --path $model --task $TASK \
    --gen-subset test -t $TGT -s $SRC --bpe 'sentencepiece' \
    --sentencepiece-model $SPM_MODEL \
    --max-sentences 32 \
     --num-workers 70 \
     --beam 15 \
     --results-path outputs \
     --skip-invalid-size-inputs-valid-test \
     --source-lang $SRC --target-lang $TGT;
}

########################
## Start of pipeline  ##
########################

echo "Preprocessing documents."
preprocess;

####################
## translation  ##
####################
CURR_TRAINSET_PATH=$TRAINSET_PATH
echo "Start self-training loop."
for I in 1
do
    echo "Training model."
    train_translate $CURR_TRAINSET_PATH $SRC $TGT $TASK;
    echo "Generating outputs."
    generate;
done
