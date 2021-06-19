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
#CORPUS_PATH=corpus/graph

#MONO_SRC=~/wikibio-xx-master/test_fr/train.ib
#MONO_SRC=corpus/graph/train.ib

# copy shared test set over
cp ~/wikibio-xx-master/testset/test.$SRC ~/wikibio-xx-master/$TGT/test.$SRC
cp ~/wikibio-xx-master/testset/test.$TGT ~/wikibio-xx-master/$TGT/test.$TGT

# create result dir
mkdir -p results
#mkdir -p $CORPUS_PATH/st_corpus

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
#ST_TRAIN_SRC_PATH=$CORPUS_PATH/st_corpus/mono.$SRC
#cp $MONO_SRC $ST_TRAIN_SRC_PATH

TRAIN_SRC_PATH=$CORPUS_PATH/train.$SRC
TEST_SRC_PATH=$CORPUS_PATH/test.$SRC
DEV_SRC_PATH=$CORPUS_PATH/valid.$SRC

TRAIN_TGT_PATH=$CORPUS_PATH/train.$TGT
TEST_TGT_PATH=$CORPUS_PATH/test.$TGT
DEV_TGT_PATH=$CORPUS_PATH/valid.$TGT

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
#build_spm_model;
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
mv $DEV_SRC_PATH.spm $CORPUS_PATH/valid.spm.$SRC
mv $DEV_TGT_PATH.spm $CORPUS_PATH/valid.spm.$TGT
mv $TEST_SRC_PATH.spm $CORPUS_PATH/test.spm.$SRC
mv $TEST_TGT_PATH.spm $CORPUS_PATH/test.spm.$TGT

preprocess() {
    python preprocess.py \
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
WARMUP_UPDATES=100;
PATIENCE=50;
TOTAL_EPOCH=200;

TASK=translation;
ARCH=transformer_wmt_en_de;


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

## Prepare Self-training
ST_OUTPUTS=self_train_outputs
ST_CORPUS=data-bin/$ST_OUTPUTS
# create result dir
mkdir -p $ST_CORPUS

# Create directories for augmentation
PARA_DATA=$(readlink -f $CORPUS_PATH/$CORPUS_DEST)
ST_DATA=$(readlink -f $ST_CORPUS)
COMB_DATA=data-bin/para_plus_st


########################
## Start of pipeline  ##
########################

echo "Preprocessing documents."
preprocess;


####################
## Self-Training  ##
####################
CURR_TRAINSET_PATH=$TRAINSET_PATH
echo "Start self-training loop."
for I in 1
do
    echo "Training model."
    train_translate $CURR_TRAINSET_PATH $SRC $TGT $TASK;
    # clean previous augmentations
    rm -f $ST_CORPUS
    #rm -f $COMB_DATA/*
    echo "Generating outputs."
    bash gen.sh $SRC $TRG 5;
    # Get BLEU-4
    #bash run_st_aug.sh $SRC $TGT
    echo "Changing train set path to...${COMB_DATA}";
    #CURR_TRAINSET_PATH=$COMB_DATA
done
