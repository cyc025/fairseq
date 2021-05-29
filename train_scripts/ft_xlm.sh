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
CORPUS_PATH=corpus/wikibio
#CORPUS_PATH=corpus/graph


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
rm -f $CORPUS_PATH/pretrain/*

# define self-training source, to be modified
ST_TRAIN_SRC_PATH=$CORPUS_PATH/st_corpus/mono.$SRC

TRAIN_SRC_PATH=$CORPUS_PATH/train.$SRC
TEST_SRC_PATH=$CORPUS_PATH/test.$SRC
DEV_SRC_PATH=$CORPUS_PATH/valid.$SRC

TRAIN_TGT_PATH=$CORPUS_PATH/train.$TGT
TEST_TGT_PATH=$CORPUS_PATH/test.$TGT
DEV_TGT_PATH=$CORPUS_PATH/valid.$TGT

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
#build_spm_model;
SPM_MODEL_PATH=xlmr_base/sentencepiece.bpe
SPM_MODEL=xlmr_base/sentencepiece.bpe.model
#SPM_MODEL_PATH=corpus/e2e/src-tgt
#SPM_MODEL=corpus/e2e/src-tgt.model

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
python sentence_piece.py --mode doc2spm \
--model_path $SPM_MODEL_PATH \
--corpus $ST_TRAIN_SRC_PATH

# rename paths for preprocessing
mv $TRAIN_SRC_PATH.spm $CORPUS_PATH/train.spm.$SRC
mv $TRAIN_TGT_PATH.spm $CORPUS_PATH/train.spm.$TGT
mv $DEV_SRC_PATH.spm $CORPUS_PATH/valid.spm.$SRC
mv $DEV_TGT_PATH.spm $CORPUS_PATH/valid.spm.$TGT
mv $TEST_SRC_PATH.spm $CORPUS_PATH/test.spm.$SRC
mv $TEST_TGT_PATH.spm $CORPUS_PATH/test.spm.$TGT


#cp xlmr_base/dict.txt corpus/e2e/ib_en/.
preprocess() {
    python preprocess.py \
    --task cross_lingual_lm \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref $CORPUS_PATH/train.spm \
    --validpref $CORPUS_PATH/valid.spm \
    --testpref $CORPUS_PATH/test.spm  \
    --destdir $CORPUS_PATH/$CORPUS_DEST \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --joined-dictionary \
    --workers 70;
}




# Model configurations
WARMUP_UPDATES=500;
PATIENCE=1000;
#WARMUP_UPDATES=10;
TOTAL_NUM_UPDATE=1000;

TASK=translation;
ARCH=transformer_wmt_en_de; #roberta_base;



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
    --save-dir checkpoints \
    --no-epoch-checkpoints \
    --no-last-checkpoints;
}

mkdir -p outputs/$TGT
model=checkpoints/$TGT/checkpoint_best.pt
generate() {
    python generate.py $CORPUS_PATH/$CORPUS_DEST \
    --path $model --task $TASK \
    --gen-subset test -t $TGT -s $SRC --bpe 'sentencepiece' \
    --sentencepiece-vocab $SPM_MODEL \
    --max-sentences 32 \
     --num-workers 70 \
     --beam 5 \
     --results-path outputs/$TGT \
     --skip-invalid-size-inputs-valid-test \
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

    mv $COMB_DATA/$SRC.spm $COMB_DATA/train.spm.$SRC;
    mv $COMB_DATA/$TGT.spm $COMB_DATA/train.spm.$TGT;

    python preprocess.py \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref $COMB_DATA/train.spm \
    --validpref $CORPUS_PATH/valid.spm \
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
#preprocess_self_training;


####################
## Self-Training  ##
####################
CURR_TRAINSET_PATH=$TRAINSET_PATH
echo "Start self-training loop."
#for I in 1 2 3
for I in 1
do
    #echo "Changing train set path to...${COMB_DATA}";
    #echo "Backward-training model."
    #train_translate $CURR_TRAINSET_PATH $TGT $SRC;
    echo "Training model."
    train_translate $CURR_TRAINSET_PATH $SRC $TGT $TASK;
    # clean previous augmentations
    #rm -f $ST_CORPUS/*
    #rm -f $COMB_DATA/*
    echo "Generating outputs."
    generate;
    #echo "Generating ST outputs."
    #generate_for_self_training;
    # Get BLEU-4
    tail -1 outputs/$TGT/generate-test.txt >> results/$SRC-$TGT.$I
    #echo "Creating self-training parallel data."
    #grep 'H-' $CORPUS_PATH/st_corpus/generate-test.txt | awk -F '\t' '{print $3}' > $ST_CORPUS/$TGT
    #grep 'S-' $CORPUS_PATH/st_corpus/generate-test.txt | awk -F '\t' '{print $2}' > $ST_CORPUS/$SRC
    #echo "Preprocessing parallel data.";
    #self_train_preprocess;
    # Set self-training dataset
    #rm -rf checkpoints/$TGT
    #CURR_TRAINSET_PATH=$COMB_DATA
done
