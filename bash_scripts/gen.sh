set -e


SRC=$1
TGT=$2


beam_size=$3

CORPUS_PATH=~/wikibio-xx-master/$TGT
CORPUS_DEST=${SRC}_${TGT}



# Model configurations
WARMUP_UPDATES=500;
PATIENCE=1000;
#WARMUP_UPDATES=10;
TOTAL_NUM_UPDATE=1000;

TASK=translation;
ARCH=transformer_wmt_en_de;

#SPM_MODEL_PATH=$CORPUS_PATH/$TGT
#SPM_MODEL=$SPM_MODEL_PATH.model
SPM_MODEL_PATH=mbart/cc25_pretrain/sentence.bpe
SPM_MODEL=mbart/cc25_pretrain/sentence.bpe.model

#mkdir -p outputs/$TGT
model=checkpoints/$SRC-$TGT/checkpoint_best.pt
generate() {
    python generate.py $CORPUS_PATH/$CORPUS_DEST \
    --path $model --task $TASK \
    --gen-subset test -t $TGT -s $SRC --bpe 'sentencepiece' \
    --sentencepiece-vocab $SPM_MODEL \
    --max-sentences 32 \
     --num-workers 70 \
     --beam $beam_size \
     --results-path outputs/$SRC-$TGT \
     --skip-invalid-size-inputs-valid-test \
     --source-lang $SRC --target-lang $TGT;
}

generate;

tail -1 outputs/$TGT/generate-test.txt | echo
