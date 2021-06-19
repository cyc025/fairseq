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
ST_TRAIN_SRC_PATH=$CORPUS_PATH/st_corpus/train.$SRC

TRAIN_SRC_PATH=$CORPUS_PATH/train.$SRC
TEST_SRC_PATH=$CORPUS_PATH/test.$SRC
DEV_SRC_PATH=$CORPUS_PATH/valid.$SRC

TRAIN_TGT_PATH=$CORPUS_PATH/train.$TGT
TEST_TGT_PATH=$CORPUS_PATH/test.$TGT
DEV_TGT_PATH=$CORPUS_PATH/valid.$TGT

# define preprocessed corpus
TRAINSET_PATH=$CORPUS_PATH/$CORPUS_DEST



################################################################################

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
build_spm_model;
#SPM_MODEL_PATH=mbart/cc25_pretrain/sentence.bpe
#SPM_MODEL=mbart/cc25_pretrain/sentence.bpe.model
SPM_MODEL_PATH=$CORPUS_PATH/$SRC-$TGT
SPM_MODEL=$CORPUS_PATH/$SRC-$TGT.model

################################################################################

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


################################################################################
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
################################################################################

# Model configurations
WARMUP_UPDATES=100;
TOTAL_NUM_UPDATE=200;
TOTAL_EPOCH=100;
TASK=multi_semisupervised_translation;
ARCH=multilingual_transformer_iwslt_de_en;


train_translate() {
    python train.py $1  --fp16 \
    --task $TASK \
    --arch $ARCH \
    --lang-pairs $SRC-$TGT \
    --ddp-backend=no_c10d \
    --max-epoch $TOTAL_EPOCH \
    --share-decoders --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --min-lr '1e-09' --lr-scheduler inverse_sqrt --warmup-updates $WARMUP_UPDATES \
    --dropout 0.3 --weight-decay 0.0001 --warmup-init-lr '1e-07' \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --keep-interval-updates 5 \
    --keep-best-checkpoints	 5 \
    --lambda-denoising-config 0.1  \
    --max-tokens 2048 --update-freq 2 ;
}

################################################################################

model=checkpoints/checkpoint_best.pt
generate() {
    python generate.py $CORPUS_PATH/$CORPUS_DEST \
    --path $model --task $TASK \
    --gen-subset test --bpe 'sentencepiece' \
    --sentencepiece-vocab $SPM_MODEL \
    --max-sentences 32 \
    --num-workers 70 \
    --beam 15 \
    --results-path outputs \
    --skip-invalid-size-inputs-valid-test \
    --lang-pairs $SRC-$TGT,$TGT-$SRC \
    --source-lang $SRC --target-lang $TGT;
}
generate_for_self_training() {
     cp $CORPUS_PATH/$CORPUS_DEST/dict.$TGT.txt $CORPUS_PATH/st_corpus/.
     python generate.py $CORPUS_PATH/st_corpus \
     --path $model --task $TASK \
     --gen-subset test --bpe 'sentencepiece' \
     --sentencepiece-vocab $SPM_MODEL \
     --max-sentences 32 \
     --num-workers 70 \
     --beam 15 \
     --results-path $CORPUS_PATH/st_corpus \
     --skip-invalid-size-inputs-valid-test \
     --lang-pairs $SRC-$TGT,$TGT-$SRC \
     --source-lang $SRC --target-lang $TGT;
}
################################################################################


########################
## Start of pipeline  ##
########################

echo "Preprocessing documents."
preprocess;

####################
## Training  ##
####################
CURR_TRAINSET_PATH=$TRAINSET_PATH

echo "Changing train set path to...${COMB_DATA}";
echo "Training model."
train_translate $CURR_TRAINSET_PATH $TASK;
# clean previous augmentations
rm -f $ST_CORPUS/*
rm -f $COMB_DATA/*
echo "Generating outputs."
generate
