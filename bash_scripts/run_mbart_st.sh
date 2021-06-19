#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Under CORPUS_PATH, create st_corpus to store monolingual source data.
# Name all files as [train|dev|test].[$SRC|$TGT]
#


set -e

SRC=src
TGT=tgt
CORPUS_PATH=~/fairseq/graph_data
MONO_SRC=~/fairseq/graph_data/train.src

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

# define self-training source, to be modified
ST_TRAIN_SRC_PATH=$CORPUS_PATH/st_corpus/mono.$SRC
cp $MONO_SRC $ST_TRAIN_SRC_PATH

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
#build_spm_model;
SPM_MODEL_PATH=mbart/cc25_pretrain/sentence.bpe
SPM_MODEL=mbart/cc25_pretrain/sentence.bpe.model
#SPM_MODEL_PATH=corpus/e2e/ib-en
#SPM_MODEL=corpus/e2e/ib-en.model

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


preprocess() {
    python preprocess.py \
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
WARMUP_UPDATES=10;
TOTAL_NUM_UPDATE=20;
TOTAL_EPOCH=100;
langs=$SRC,$TGT;
TASK=translation_from_pretrained_bart;
ARCH=mbart_large;
train_translate() {
    python train.py $1 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --arch $ARCH \
    --task $TASK \
    --max-epoch $TOTAL_EPOCH \
    --source-lang $SRC --target-lang $TGT \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.2  --dataset-impl mmap \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay --lr 3e-05 \
    --min-lr -1 --warmup-updates $WARMUP_UPDATES \
    --total-num-update $TOTAL_NUM_UPDATE --dropout 0.3 \
    --attention-dropout 0.1  --weight-decay 0.0 \
    --max-tokens 1024 --update-freq 2 --save-interval 1 \
    --save-interval-updates 50 --keep-interval-updates 5 \
    --no-epoch-checkpoints --seed 222 --log-format simple \
    --log-interval 2 --reset-optimizer --reset-meters \
    --encoder-layers 3 --decoder-layers 3 \
    --reset-dataloader --reset-lr-scheduler \
    --langs $langs --layernorm-embedding  --ddp-backend no_c10d;
}

model=checkpoints/checkpoint_best.pt
generate() {
    python generate.py $CORPUS_PATH/$CORPUS_DEST  \
     --path $model  --task $TASK \
     --gen-subset test -t $TGT -s $SRC --bpe 'sentencepiece' \
     --sentencepiece-vocab $SPM_MODEL \
     --remove-bpe 'sentencepiece' --max-sentences 32 \
     --results-path outputs \
     --skip-invalid-size-inputs-valid-test \
     --langs $langs;

}
generate_for_self_training() {
     cp $CORPUS_PATH/$CORPUS_DEST/dict.$TGT.txt $CORPUS_PATH/st_corpus/.
     python generate.py $CORPUS_PATH/st_corpus  \
      --path $model  --task $TASK \
      --gen-subset test -t $TGT -s $SRC --bpe 'sentencepiece' \
      --sentencepiece-vocab $SPM_MODEL \
      --remove-bpe 'sentencepiece' --max-sentences 32 \
      --results-path $CORPUS_PATH/st_corpus \
      --skip-invalid-size-inputs-valid-test \
      --langs $langs;
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

    mv $COMB_DATA/$SRC.spm $COMB_DATA/train.spm.$SRC;
    mv $COMB_DATA/$TGT.spm $COMB_DATA/train.spm.$TGT;

    python preprocess.py \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref $COMB_DATA/train.spm \
    --validpref $CORPUS_PATH/dev.spm \
    --testpref $CORPUS_PATH/test.spm  \
    --destdir ${COMB_DATA} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --joined-dictionary \
    --workers 70;
}

########################
## Start of pipeline  ##
########################

echo "Preprocessing documents."
preprocess;
echo "Preprocessing self-training documents."
preprocess_self_training;


####################
## Self-Training  ##
####################
CURR_TRAINSET_PATH=$TRAINSET_PATH
echo "Start self-training loop."
for I in 1 2 3
do
    echo "Changing train set path to...";
    echo $COMB_DATA;
    #echo "Backward-training model."
    #train_translate $CURR_TRAINSET_PATH $TGT $SRC;
    echo "Training model."
    train_translate $CURR_TRAINSET_PATH $SRC $TGT;
    # clean previous augmentations
    rm -f $ST_CORPUS/*
    rm -f $COMB_DATA/*
    echo "Generating outputs."
    generate;
    # echo "Generating ST outputs."
    # generate_for_self_training;
    # # Get BLEU-4
    # tail -1 outputs/generate-test.txt >> results/$SRC-$TGT.$I
    # echo "Creating self-training parallel data."
    # grep 'H-' $CORPUS_PATH/st_corpus/generate-test.txt | awk -F '\t' '{print $3}' > $ST_CORPUS/$TGT
    # grep 'S-' $CORPUS_PATH/st_corpus/generate-test.txt | awk -F '\t' '{print $2}' > $ST_CORPUS/$SRC
    # echo "Preprocessing parallel data.";
    # self_train_preprocess;
    # # Set self-training dataset
    # rm -f checkpoints/*
    # CURR_TRAINSET_PATH=$COMB_DATA
done
