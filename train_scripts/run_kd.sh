#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This script utilize pretrained model to generate training data.
# Then Train a new model on this pseudo parallel data.

# Under CORPUS_PATH, create st_corpus to store monolingual source data.
# Name all files as [train|dev|test].[$SRC|$TGT]
#


set -e


SRC=en
TGT=fr
CORPUS_PATH=en_fr
TASK=translation_from_pretrained_bart
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN


CORPUS_DEST=${SRC}_${TGT}

# clean files
rm -f $CORPUS_PATH/$CORPUS_DEST/*
rm -f $CORPUS_PATH/*spm*
rm -f $CORPUS_PATH/*.vocab
rm -f $CORPUS_PATH/*.model
rm -f $CORPUS_PATH/*.bin
rm -f $CORPUS_PATH/*.idx
rm -f $CORPUS_PATH/*.txt
rm -f results/*

# create result dir
mkdir -p results

TRAIN_SRC_PATH=$CORPUS_PATH/train.$SRC
TEST_SRC_PATH=$CORPUS_PATH/test.$SRC
DEV_SRC_PATH=$CORPUS_PATH/dev.$SRC

TRAIN_TGT_PATH=$CORPUS_PATH/train.$TGT
TEST_TGT_PATH=$CORPUS_PATH/test.$TGT
DEV_TGT_PATH=$CORPUS_PATH/dev.$TGT

# define preprocessed corpus
TRAINSET_PATH=$CORPUS_PATH/$CORPUS_DEST

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

preprocess_KD() {
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
train() {
    # Model configurations
    WARMUP_UPDATES=2500;
    TOTAL_NUM_UPDATE=40000;
    python train.py $TRAINSET_PATH  \
    --arch mbart_large --task $TASK  \
    --langs $langs \
    --criterion label_smoothed_cross_entropy --fp16 \
    --label-smoothing 0.2  --dataset-impl mmap \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 \
    --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 \
    --max-tokens 2048 --update-freq 2 --save-interval 1 \
    --no-epoch-checkpoints --seed 222 --log-format simple \
    --log-interval 2 --reset-optimizer --reset-meters \
    --reset-dataloader --reset-lr-scheduler \
    --layernorm-embedding  --ddp-backend no_c10d \
    --maximize-best-checkpoint-metric \
    --arch mbart_large \
    --no-last-checkpoints \
    --warmup-updates $WARMUP_UPDATES \
    --total-num-update $TOTAL_NUM_UPDATE \
    --save-interval-updates 20 \
    --keep-interval-updates 5 \
    --encoder-layers 4 --decoder-layers 4;
    #--arch multilingual_transformer \
    #--lang-pairs $SRC-$TGT,$TGT-$SRC;
}
model=checkpoints/checkpoint_best.pt
generate_for_KD() {
    cp $CORPUS_PATH/$CORPUS_DEST/dict.$TGT.txt $CORPUS_PATH/st_corpus/.
    python generate.py $CORPUS_PATH/st_corpus \
    --path $model --task $TASK \
    --gen-subset test -t $TGT -s $SRC --bpe 'sentencepiece' \
    --sentencepiece-vocab $SPM_MODEL \
    --max-sentences 32 --iter-decode-force-max-iter \
     --num-workers 70 \
     --langs $langs \
     --results-path $CORPUS_PATH/st_corpus --beam 15;
     #--lang-pairs $SRC-$TGT,$TGT-$SRC;
}

########################
## Start of pipeline  ##
########################

echo "Preprocessing documents."
preprocess;
echo "Generating KD outputs."
generate_for_KD;
# Get BLEU-4
tail -1 outputs/generate-test.txt >> results/$SRC-$TGT.$I
echo "Creating self-training parallel data."
grep 'H-' $CORPUS_PATH/st_corpus/generate-test.txt | awk -F '\t' '{print $3}' > $ST_CORPUS/$TGT
grep 'S-' $CORPUS_PATH/st_corpus/generate-test.txt | awk -F '\t' '{print $3}' > $ST_CORPUS/$SRC
echo "Preprocessing documents."
preprocess_KD;
echo "Train on pseudo-data."
train;
