#!/bin/bash


data_dir=data-bin/iwslt14.tokenized.de-en
save_path=saved_models




# CUDA_VISIBLE_DEVICES=7 python3 train.py ${data_dir} \
#     --save-dir ${save_path} \
#     --arch transformer_wmt_en_de --share-decoder-input-output-embed \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#     --dropout 0.3 --weight-decay 0.0001 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --max-tokens 4096 \
#     --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512  \
#     --eval-bleu --no-epoch-checkpoints \
#     --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
#     --eval-bleu-detok moses \
#     --eval-bleu-remove-bpe \
#     --eval-bleu-print-samples \
#     --skip-invalid-size-inputs-valid-test \
#     --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
#     --save-interval-updates 30000 \
#     --max-update 10000  \

# 6-6-512: 35
# 12-1-512: 27.24
# 11-2-512: 29.38
# 12-1-64:


python3 train.py ${data_dir} \
    --save-dir ${save_path} --fp16 \
    --arch transformer_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --encoder-layers 12 --encoder-embed-dim 512 --decoder-layers 1 --decoder-embed-dim 64  \
    --eval-bleu --no-epoch-checkpoints \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-interval-updates 30000 \
    --max-update 10000000  \
