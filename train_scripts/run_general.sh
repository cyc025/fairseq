#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Under CORPUS_PATH, create st_corpus to store monolingual source data.
# Name all files as [train|dev|test].[$SRC|$TGT]
#


set -e


SRC=ib
TGT=en
CORPUS_PATH=corpus/e2e_10


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



TRAIN_SRC_PATH=$CORPUS_PATH/train.$SRC
TEST_SRC_PATH=$CORPUS_PATH/test.$SRC
DEV_SRC_PATH=$CORPUS_PATH/dev.$SRC

TRAIN_TGT_PATH=$CORPUS_PATH/train.$TGT
TEST_TGT_PATH=$CORPUS_PATH/test.$TGT
DEV_TGT_PATH=$CORPUS_PATH/dev.$TGT

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
#SPM_MODEL_PATH=mbart/cc25_pretrain/sentence.bpe
#SPM_MODEL=mbart/cc25_pretrain/sentence.bpe.model
SPM_MODEL_PATH=corpus/e2e/ib-en
SPM_MODEL=corpus/e2e/ib-en.model

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


# Model configurations
WARMUP_UPDATES=2500;
TOTAL_NUM_UPDATE=5000;
TASK=translation;
ARCH=transformer_wmt_en_de;

train_translate() {
    python train.py $TRAINSET_PATH  \
    --task $TASK \
    --arch $ARCH \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates $WARMUP_UPDATES \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2048 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --log-interval 2 \
    --keep-interval-updates 5 \
    --encoder-layers 3 --decoder-layers 3 \
    --skip-invalid-size-inputs-valid-test \
    --maximize-best-checkpoint-metric \
    --source-lang $SRC --target-lang $TGT;
}

train() {
    # Model configurations
    WARMUP_UPDATES=2500;
    TOTAL_NUM_UPDATE=5000;
    TASK=multilingual_translation;
    ARCH=multilingual_transformer_iwslt_de_en;
    python train.py $TRAINSET_PATH  \
    --task $TASK  --arch $ARCH \
    --criterion label_smoothed_cross_entropy --fp16 \
    --label-smoothing 0.2  --dataset-impl mmap \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 \
    --dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 \
    --max-tokens 2048 --update-freq 2 \
    --no-epoch-checkpoints --seed 222 --log-format simple \
    --log-interval 2 --reset-optimizer --reset-meters \
    --reset-dataloader --reset-lr-scheduler \
    --layernorm-embedding  --ddp-backend no_c10d \
    --maximize-best-checkpoint-metric \
    --no-last-checkpoints \
    --warmup-updates $WARMUP_UPDATES \
    --total-num-update $TOTAL_NUM_UPDATE \
    --save-interval-updates 1000 --keep-interval-updates 5 --patience 1000 \
    --encoder-layers 3 --decoder-layers 3 \
    --source-lang $SRC --target-lang $TGT \
    --lang-pairs $SRC,$TGT;
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
model=checkpoints/checkpoint_best.pt
semi_generate() {
    python generate.py $CORPUS_PATH/$CORPUS_DEST \
    --path $model --task $TASK \
    --gen-subset test -s $SRC -t $TGT --bpe 'sentencepiece' \
    --sentencepiece-vocab $SPM_MODEL \
    --max-sentences 32 \
     --num-workers 70 \
     --beam 15 \
     --lang-pairs $SRC-$TGT,$TGT-$SRC \
     --skip-invalid-size-inputs-valid-test \
     --results-path outputs;
}


########################
## Start of pipeline  ##
########################

echo "Preprocessing documents."
preprocess;
echo "Training model."
train_translate;
echo "Generating outputs."
generate;
