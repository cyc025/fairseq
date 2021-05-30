
import sys
import argparse

import sentencepiece as spm



"""

# without dictionary
DICT=mbart/cc25_pretrain/dict.txt
python preprocess.py \
--source-lang ib \
--target-lang en \
--trainpref corpus/wikibio/train.spm \
--validpref corpus/wikibio/train.spm \
--testpref corpus/wikibio/train.spm  \
--destdir corpus/wikibio \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT} \
--tgtdict ${DICT} \
--workers 70

mv train.box.tok.spm train.spm.src
mv train.sent.tok.spm train.spm.tgt
mv test.box.tok.spm test.spm.src
mv test.sent.en.tok.spm test.spm.tgt


MODEL=mbart/cc25_pretrain/sentence.bpe.model
DICT=mbart/cc25_pretrain/dict.txt
python preprocess.py \
--source-lang en \
--target-lang fr \
--trainpref ../nlg-engine/data/current/wikibio-x/mnmt/mbart/train.spm \
--validpref ../nlg-engine/data/current/wikibio-x/mnmt/mbart/test.spm \
--testpref ../nlg-engine/data/current/wikibio-x/mnmt/mbart/test.spm  \
--destdir en_fr \
--thresholdtgt 0 \
--thresholdsrc 0 \
--srcdict ${DICT} \
--tgtdict ${DICT} \
--workers 70







PRETRAIN=mbart/cc25_pretrain


langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

python train.py en_fr  \
--encoder-normalize-before \
--decoder-normalize-before  \
--arch mbart_large --task translation_from_pretrained_bart  \
--source-lang en --target-lang fr \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.2  --dataset-impl mmap \
--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 \
--warmup-updates 2500 --total-num-update 40000 \
--dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 \
--max-tokens 1024 --update-freq 2 --save-interval 1 \
--save-interval-updates 5000 --keep-interval-updates 10 \
--no-epoch-checkpoints --seed 222 --log-format simple \
--log-interval 2 --reset-optimizer --reset-meters \
--reset-dataloader --reset-lr-scheduler --restore-file $PRETRAIN \
--langs $langs --layernorm-embedding  --ddp-backend no_c10d \
--validate-interval 1 --patience 10


# train from scratch
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
python train.py en_fr  \
--encoder-normalize-before \
--decoder-normalize-before  \
--arch mbart_large --task translation_from_pretrained_bart  \
--source-lang en --target-lang fr \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.2  --dataset-impl mmap \
--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 \
--warmup-updates 2500 --total-num-update 40000 \
--dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 \
--max-tokens 1024 --update-freq 2 --save-interval 1 \
--save-interval-updates 5000 --keep-interval-updates 10 \
--no-epoch-checkpoints --seed 222 --log-format simple \
--log-interval 2 --reset-optimizer --reset-meters \
--reset-dataloader --reset-lr-scheduler \
--langs $langs --layernorm-embedding  --ddp-backend no_c10d \
--validate-interval 1 --patience 10 \
--encoder-layers 2 --decoder-layers 2


langs=en_XX,XX_en
python train.py corpus/wikibio  \
--arch mbart_large --task translation  \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.2  --dataset-impl mmap \
--optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
--lr-scheduler polynomial_decay --lr 3e-05 --min-lr -1 \
--warmup-updates 2500 --total-num-update 40000 \
--dropout 0.3 --attention-dropout 0.1  --weight-decay 0.0 \
--max-tokens 1024 --update-freq 2 --save-interval 1 \
--save-interval-updates 5000 --keep-interval-updates 10 \
--no-epoch-checkpoints --seed 222 --log-format simple \
--log-interval 2 --reset-optimizer --reset-meters \
--reset-dataloader --reset-lr-scheduler \
--layernorm-embedding  --ddp-backend no_c10d \
--validate-interval 1 --patience 10 \
--encoder-layers 2 --decoder-layers 2






# workable version
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
model=checkpoints/checkpoint_best.pt
python generate.py en_fr --path $model \
--task translation_from_pretrained_bart \
--gen-subset test -t fr -s en --bpe 'sentencepiece' \
--sentencepiece-vocab mbart/cc25_pretrain/sentence.bpe.model  --max-sentences 32 \
--langs $langs --iter-decode-force-max-iter --num-workers 70 \
--results-path outputs --beam 1 --match-source-len --no-repeat-ngram-size 3




# workable version
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
model=checkpoints/checkpoint_best.pt
python generate.py test_set --path $model \
--task translation_from_pretrained_bart \
--gen-subset test -t fr -s en --bpe 'sentencepiece' \
--sentencepiece-vocab mbart/cc25_pretrain/sentence.bpe.model  --max-sentences 32 \
--langs $langs --iter-decode-force-max-iter --num-workers 70 \
--results-path outputs --beam 1 --match-source-len --no-repeat-ngram-size 3






Example usage (for French):


# tokenize

cat zh/sent.en | \
./mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en | \
./mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -no-escape -threads 1 \
> zh/sent.en.tok


cat sent.fr | \
./mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l fr | \
./mosesdecoder/scripts/tokenizer/tokenizer.perl -l fr -no-escape -threads 1 \
> sent.fr.tok


cat fr/pretrain.sent.fr | \
./mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l fr | \
./mosesdecoder/scripts/tokenizer/tokenizer.perl -l fr -no-escape -threads 1 \
> fr/pretrain.sent.fr.tok


cat fr/750_translations | \
./mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l fr | \
./mosesdecoder/scripts/tokenizer/tokenizer.perl -l fr -no-escape -threads 1 \
> fr/750_translations.tok

cat fr/train.box | \
./mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en | \
./mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -no-escape -threads 1 \
> fr/train.box.tok




cat sample_source.summary.10.gpt.nodup | \
~/nlg-engine/data/current/wikibio-x/mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l en | \
~/nlg-engine/data/current/wikibio-x/mosesdecoder/scripts/tokenizer/tokenizer.perl -l en -no-escape -threads 1 \
> sample_source.summary.10.gpt.nodup.tok




# create dataset

head -10000 sent.en.split.tok > train.sent.en
awk 'FNR>=10001 && FNR<=11250' sent.en.split.tok > dev.sent.en
awk 'FNR>=11251 && FNR<=12500' sent.en.split.tok > test.sent.en


# build - for pretrained dataset
python utils/sentence_piece.py --mode build \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/pretrain.sent.en.tok

python utils/sentence_piece.py --mode build \
--model_path embeddings/wikibio/fr \
--corpus data/current/wikibio-x/fr/pretrain.sent.fr.tok

# convert to ids - for pretrained dataset

python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/pretrain.sent.en.tok

python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/test.sent.en.tok

python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr \
--corpus data/current/wikibio-x/fr/pretrain.sent.fr.tok




python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/train.sent.box.en.tok




./utils/fastText-0.2.0/fasttext skipgram \
-input data/current/wikibio-x/fr/pretrain.sent.en.tok.spm \
-output embeddings/wikibio/fr-en -minCount 0 -dim 1

./utils/fastText-0.2.0/fasttext skipgram \
-input data/current/wikibio-x/fr/pretrain.sent.fr.tok.spm \
-output embeddings/wikibio/fr -minCount 0 -dim 1






python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/de.sent.en.tok






#### regualr pipelines


# build for only parallel


python utils/sentence_piece.py --mode build \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/6k.train.sent.en.tok

python utils/sentence_piece.py --mode build \
--model_path embeddings/wikibio/fr \
--corpus data/current/wikibio-x/fr/6k.train.sent.fr.tok

python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/6k.train.sent.en.tok

python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr \
--corpus data/current/wikibio-x/fr/6k.train.sent.fr.tok

python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/test.sent.en.tok



python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/test.box.tok


python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/train.box.tok



./utils/fastText-0.2.0/fasttext skipgram \
-input data/current/wikibio-x/fr/6k.train.sent.en.tok.spm \
-output embeddings/wikibio/fr-en -minCount 0 -dim 1

./utils/fastText-0.2.0/fasttext skipgram \
-input data/current/wikibio-x/fr/6k.train.sent.fr.tok.spm \
-output embeddings/wikibio/fr -minCount 0 -dim 1


./utils/fastText-0.2.0/fasttext skipgram \
-input data/current/wikibio-x/fr/train.box \
-output embeddings/wikibio/box -minCount 0 -dim 1


###############################################################

# build for combined data
python utils/sentence_piece.py --mode build \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/com.sent.en.tok

python utils/sentence_piece.py --mode build \
--model_path embeddings/wikibio/fr \
--corpus data/current/wikibio-x/fr/com.sent.fr.tok


# convert to ids
python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/com.sent.en.tok

python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr \
--corpus data/current/wikibio-x/fr/com.sent.fr.tok

python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/test.sent.en.tok






python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/6k.train.box.sent.en.tok

python utils/sentence_piece.py --mode doc2ids \
--model_path embeddings/wikibio/fr-en \
--corpus data/current/wikibio-x/fr/test.box.sent.en.tok









./utils/fastText-0.2.0/fasttext skipgram \
-input data/current/wikibio-x/fr/com.sent.en.tok.spm \
-output embeddings/wikibio/fr-en -minCount 0 -dim 1

./utils/fastText-0.2.0/fasttext skipgram \
-input data/current/wikibio-x/fr/com.sent.fr.tok.spm \
-output embeddings/wikibio/fr -minCount 0 -dim 1






"""





# Build argument parser
parser = argparse.ArgumentParser(description='Train sentence piece model')
parser.add_argument('--mode', default='ids2doc', help='specify mode of use')
parser.add_argument('--model_path', help='the path to model')
parser.add_argument('--corpus', help='the corpus')
parser.add_argument('--vocab_size', type=int, default=6000, help='the evaluation threshold ')

args = parser.parse_args()


def build_piece_dict():
    spm.SentencePieceTrainer.train(
        '--input={0} \
        --model_prefix={1} \
        --vocab_size={2} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 \
        --hard_vocab_limit=false'.format(args.corpus,args.model_path,args.vocab_size)
    )


def doc2ids():

    corpus = open(args.corpus,'r').readlines()
    new_corpus = open('{0}.spm'.format(args.corpus),'w')

    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(args.model_path))

    for line in corpus:
        new_corpus.write(' '.join([str(id) for id in sp.encode_as_ids(line)])+'\n')
    new_corpus.close()


def ids2doc():

    corpus = open(args.corpus,'r').readlines()
    new_corpus = open('{0}.txt'.format(args.corpus),'w')

    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(args.model_path))

    for line in corpus:
        line = line.replace('<unk>','').strip()
        new_corpus.write(''.join(sp.decode_ids( [ int(id) for id in line.split()] ))+'\n')
    new_corpus.close()



def doc2spm():

    corpus = open(args.corpus,'r').readlines()
    new_corpus = open('{0}.spm'.format(args.corpus),'w')

    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(args.model_path))

    for line in corpus:
        new_corpus.write(' '.join([str(id) for id in sp.EncodeAsPieces(line)])+'\n')
    new_corpus.close()


def spm2doc():

    corpus = open(args.corpus,'r').readlines()
    new_corpus = open('{0}.txt'.format(args.corpus),'w')

    sp = spm.SentencePieceProcessor()
    sp.load('{}.model'.format(args.model_path))

    for line in corpus:
        line = line.replace('<unk>','').strip()
        new_corpus.write(sp.DecodePieces( [ spm for spm in line.split()])+'\n')
    new_corpus.close()


if __name__ == '__main__':
    if args.mode == 'ids2doc':
        ids2doc()
    elif args.mode == 'doc2ids':
        doc2ids()
    elif args.mode == 'spm2doc':
        spm2doc()
    elif args.mode == 'doc2spm':
        doc2spm()
    elif args.mode == 'build':
        build_piece_dict()
