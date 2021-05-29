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
#CORPUS_PATH=~/wikibio-xx-master/test_fr
CORPUS_PATH=~/wikibio-xx-master/$TGT
#CORPUS_PATH=corpus/graph

MONO_SRC=~/wikibio-xx-master/test_fr/new_train.ib
#MONO_SRC=corpus/graph/train.ib


# create result dir
mkdir -p results
mkdir -p $CORPUS_PATH/st_corpus

CORPUS_DEST=${SRC}_${TGT}

# clean files


# define self-training source, to be modified
ST_TRAIN_SRC_PATH=$CORPUS_PATH/st_corpus/mono.$SRC
cp $MONO_SRC $ST_TRAIN_SRC_PATH

TRAIN_SRC_PATH=$CORPUS_PATH/new_train.$SRC.tok
TEST_SRC_PATH=$CORPUS_PATH/new_test.$SRC.tok
DEV_SRC_PATH=$CORPUS_PATH/new_valid.$SRC.tok

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
#build_spm_model;
SPM_MODEL_PATH=mbart/cc25_pretrain/sentence.bpe
SPM_MODEL=mbart/cc25_pretrain/sentence.bpe.model
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
mv $TRAIN_SRC_PATH.spm $CORPUS_PATH/new_train.spm.$SRC
mv $TRAIN_TGT_PATH.spm $CORPUS_PATH/new_train.spm.$TGT
mv $DEV_SRC_PATH.spm $CORPUS_PATH/new_valid.spm.$SRC
mv $DEV_TGT_PATH.spm $CORPUS_PATH/new_valid.spm.$TGT
mv $TEST_SRC_PATH.spm $CORPUS_PATH/new_test.spm.$SRC
mv $TEST_TGT_PATH.spm $CORPUS_PATH/new_test.spm.$TGT

preprocess() {
    DICT=mbart/cc25_pretrain/dict.txt
    python preprocess.py \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref $CORPUS_PATH/new_train.spm \
    --validpref $CORPUS_PATH/new_valid.spm \
    --testpref $CORPUS_PATH/new_test.spm  \
    --destdir $CORPUS_PATH/$CORPUS_DEST \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --srcdict ${DICT} \
    --tgtdict ${DICT} \
    --workers 70;
}

# Model configurations
PRETRAIN=mbart/cc25_pretrain/model.pt
WARMUP_UPDATES=1000;
TOTAL_NUM_UPDATE=2000;
langs=$SRC,$TGT;
TASK=translation_from_pretrained_bart;
ARCH=mbart_large;
train_translate() {
    python train.py $1 \
    --encoder-normalize-before --decoder-normalize-before \
    --arch mbart_large --layernorm-embedding \
    --task translation_from_pretrained_bart \
    --source-lang $SRC --target-lang $TGT \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --dataset-impl mmap \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 --warmup-updates 2500 --total-num-update 40000 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
    --max-tokens 1024 --update-freq 2 \
    --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
    --seed 222 --log-format simple --log-interval 2 \
    --restore-file $PRETRAIN \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
    --langs $langs \
    --ddp-backend no_c10d;
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


########################
## Start of pipeline  ##
########################

echo "Preprocessing documents."
#preprocess;

####################
## Self-Training  ##
####################
CURR_TRAINSET_PATH=$TRAINSET_PATH
echo "Start training loop."
for I in 1
do
    echo "Changing train set path to...";
    echo "Training model."
    train_translate $CURR_TRAINSET_PATH $SRC $TGT;
    echo "Generating outputs."
    generate;
    # Get BLEU-4
    tail -1 outputs/generate-test.txt >> results/$SRC-$TGT.$I
done
