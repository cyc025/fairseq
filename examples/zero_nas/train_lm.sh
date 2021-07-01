#!/bin/bash

# upperlim=2
# for ((i=2; i<=upperlim; i++)); do
#   echo $i > .max.len
#   taskset --cpu-list 1 python examples/bart/summarize.py   --model-dir .   --model-file checkpoints/bart.large.cnn/model.pt   --src ~/fairseq_cnn_data/cnn_cln/toy_test.source   --out ~/fairseq_cnn_data/cnn_cln/test.hypo;
#   rm .max.len
# done

# shopt -s expand_aliases;

# to be deleted
# alias with-proxy='env http_proxy=fwdproxy:8080 https_proxy=fwdproxy:8080 no_proxy=.fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost RSYNC_PROXY=fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080'

# upperlim=100
# startlen=0
# for ((i=startlen; i<=upperlim; i++)); do
#     for ((j=1; j<=10; j++)); do
#         echo $i > .curr_index
#         echo $i > .max.len
#         # taskset --cpu-list 1 python examples/bart/summarize.py   --model-dir data-bin/cnn_dm --model-file ~/checkpoints/bart.large.cnn/model.pt   --src data-bin/cnn_dm/toy_test.source   --out data-bin/cnn_dm/test.hypo;
#         taskset --cpu-list 1 python examples/bart/summarize.py   --model-dir .   --model-file checkpoints/bart.base/model.pt   --src ~/fairseq_cnn_data/cnn_cln/toy_test.source   --out ~/fairseq_cnn_data/cnn_cln/test.hypo;
#         rm .max.len .curr_index
#     done
# done

TEXT=examples/zero_nas/wikitext-2
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-2 \
    --workers 20


python train.py --task language_modeling \
  data-bin/wikitext-103 \
  --save-dir checkpoints/transformer_wikitext-103 \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 2048 --update-freq 16 \
  --fp16 \
  --max-update 50000
