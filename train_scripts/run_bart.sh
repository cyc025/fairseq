#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Under CORPUS_PATH, create st_corpus to store monolingual source data.
# Name all files as [train|dev|test].[$SRC|$TGT]
#


set -e



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
TASK=translation_from_pretrained_bart;
ARCH=bart_base;
#
# TASK=translation;
# ARCH=transformer_wmt_en_de;

train_translate() {
    python train.py $1  \
    --max-tokens 4400 \
    --task $TASK \
    --add-prev-output-tokens \
    --layernorm-embedding \
    --share-all-embeddings \
    --warmup-updates $WARMUP_UPDATES
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --patience $PATIENCE \
    --arch bart_large \
    --criterion nat_loss \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 5e-4 --total-num-update $TOTAL_NUM_UPDATE \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch 10 \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric;
}


model=checkpoints/checkpoint_best.pt
generate() {
    python generate.py $CORPUS_PATH/$CORPUS_DEST \
    --path $model --task $TASK \
    --gen-subset test -t $TGT -s $SRC --bpe 'sentencepiece' \
    --sentencepiece-vocab $SPM_MODEL \
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
