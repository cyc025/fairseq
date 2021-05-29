#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Under CORPUS_PATH, create st_corpus to store monolingual source data.
# Name all files as [train|dev|test].[$SRC|$TGT]
#


set -e


SRC=en
TGT=fr
CORPUS_PATH=en_fr


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

TRAIN_SRC_PATH=$CORPUS_PATH/train.$SRC
TEST_SRC_PATH=$CORPUS_PATH/test.$SRC
DEV_SRC_PATH=$CORPUS_PATH/dev.$SRC

TRAIN_TGT_PATH=$CORPUS_PATH/train.$TGT
TEST_TGT_PATH=$CORPUS_PATH/test.$TGT
DEV_TGT_PATH=$CORPUS_PATH/dev.$TGT

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
build_spm_model;
#SPM_MODEL_PATH=mbart/cc25_pretrain/sentence.bpe
#SPM_MODEL=mbart/cc25_pretrain/sentence.bpe.model

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
WARMUP_UPDATES=2500;
TOTAL_NUM_UPDATE=5000;
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN;
TASK=translation;
ARCH=transformer_wmt_en_de;
#TASK=translation;
#ARCH=transformer;
#multilingual_transformer

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
    --encoder-layers 4 --decoder-layers 4 \
    --maximize-best-checkpoint-metric;
}

train() {
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
    --encoder-layers 4 --decoder-layers 4 \
    --source-lang $SRC --target-lang $TGT;
    #--source-lang $SRC --target-lang $TGT \
    #--lang-pairs $SRC-$TGT,$TGT-$SRC;
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
     --source-lang $SRC --target-lang $TGT;
}
generate_for_self_training() {
    cp $CORPUS_PATH/$CORPUS_DEST/dict.$TGT.txt $CORPUS_PATH/st_corpus/.
    python generate.py $CORPUS_PATH/st_corpus \
    --path $model --task $TASK \
    --gen-subset test -t $TGT -s $SRC --bpe 'sentencepiece' \
    --sentencepiece-vocab $SPM_MODEL \
    --max-sentences 32 \
     --num-workers 70 \
     --source-lang $SRC --target-lang $TGT \
     --results-path $CORPUS_PATH/st_corpus --beam 15;
}

## Prepare Self-training
ST_OUTPUTS=self_train_outputs
ST_CORPUS=data-bin/$ST_OUTPUTS
# create result dir
mkdir -p $ST_CORPUS

# 1. convert source file to .spm
# 2. preprocess both source and target spm files
self_train_preprocess() {

    python sentence_piece.py --mode doc2spm \
    --model_path $SPM_MODEL_PATH \
    --corpus $ST_CORPUS/$SRC;

    mv $ST_CORPUS/$SRC.spm $ST_CORPUS/train.spm.$SRC;
    mv $ST_CORPUS/$TGT $ST_CORPUS/train.spm.$TGT;

    python preprocess.py \
    --source-lang ${SRC} \
    --target-lang ${TGT} \
    --trainpref $ST_CORPUS/train.spm \
    --destdir ${ST_CORPUS} \
    --thresholdtgt 0 \
    --thresholdsrc 0 \
    --joined-dictionary \
    --workers 70;
}

# Create directories for augmentation
PARA_DATA=$(readlink -f $CORPUS_PATH/$CORPUS_DEST)
ST_DATA=$(readlink -f $ST_CORPUS)
COMB_DATA=data-bin/para_plus_st
mkdir -p $COMB_DATA

augment_dataset() {
    # We want to train on the combined data, so we'll symlink the parallel + ST data
    # in the wmt18_en_de_para_plus_bt directory. We link the parallel data as "train"
    # and the BT data as "train1", so that fairseq will combine them automatically
    # and so that we can use the `--upsample-primary` option to upsample the
    # parallel data (if desired).
    for LANG in $SRC $TGT; do \
        ln -s ${PARA_DATA}/dict.$LANG.txt ${COMB_DATA}/dict.$LANG.txt; \
        for EXT in bin idx; do \
            ln -s ${PARA_DATA}/train.$SRC-$TGT.$LANG.$EXT ${COMB_DATA}/train.$SRC-$TGT.$LANG.$EXT; \
            ln -s ${BT_DATA}/train.$SRC-$TGT.$LANG.$EXT ${COMB_DATA}/train1.$SRC-$TGT.$LANG.$EXT; \
            ln -s ${PARA_DATA}/valid.$SRC-$TGT.$LANG.$EXT ${COMB_DATA}/valid.$SRC-$TGT.$LANG.$EXT; \
            ln -s ${PARA_DATA}/test.$SRC-$TGT.$LANG.$EXT ${COMB_DATA}/test.$SRC-$TGT.$LANG.$EXT; \
        done; \
    done
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
echo "Start self-training loop."
for I in 1 2 3
do
    echo "Training model."
    train_translate;
    # clean previous augmentations
    rm -f $ST_CORPUS/*
    rm -f $COMB_DATA/*
    echo "Generating outputs."
    generate;
    echo "Generating ST outputs."
    generate_for_self_training;
    # Get BLEU-4
    tail -1 outputs/generate-test.txt >> results/$SRC-$TGT.$I
    echo "Creating self-training parallel data."
    grep 'H-' $CORPUS_PATH/st_corpus/generate-test.txt | awk -F '\t' '{print $3}' > $ST_CORPUS/$TGT
    grep 'S-' $CORPUS_PATH/st_corpus/generate-test.txt | awk -F '\t' '{print $3}' > $ST_CORPUS/$SRC
    echo "Preprocessing parallel data."
    self_train_preprocess;
    augment_dataset;
    # Set self-training dataset
    TRAINSET_PATH=$COMB_DATA;
done
