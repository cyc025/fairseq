#!/bin/bash


data_dir=data_bin
save_path=saved_models



# CUDA_VISIBLE_DEVICES=7 python3 train.py ${data_dir} --arch glat --noise full_mask --share-all-embeddings \
#     --criterion glat_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
#     --lr-scheduler inverse_sqrt --warmup-updates 100 --optimizer adam --adam-betas '(0.9, 0.999)' \
#     --adam-eps 1e-6 --task translation_lev_modified --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
#     --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
#     --max-source-positions 1000 --max-target-positions 1000 --max-update 300000 --seed 0 --clip-norm 5 \
#     --save-dir ${save_path} --src-embedding-copy --pred-length-offset --log-interval 5000 \
#     --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
#     --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
#     --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
#     --apply-bert-init --activation-fn gelu --user-dir glat_plugins --max-epoch 10 \
#     --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp --save-interval-updates 5000 \


# temporary
# CUDA_VISIBLE_DEVICES=7 python3 train.py ${data_dir} --arch glat --noise full_mask --share-all-embeddings \
#     --criterion glat_loss --label-smoothing 0.1 --lr 5e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
#     --lr-scheduler inverse_sqrt --warmup-updates 100 --optimizer adam --adam-betas '(0.9, 0.999)' \
#     --adam-eps 1e-6 --task translation_lev_modified --max-tokens 8192 --weight-decay 0.01 --dropout 0.1 \
#     --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 --fp16 \
#     --max-source-positions 1000 --max-target-positions 1000 --max-update 300000 --seed 0 --clip-norm 5 \
#     --save-dir ${save_path} --src-embedding-copy --pred-length-offset --log-interval 30000 \
#     --eval-bleu --eval-bleu-args '{"iter_decode_max_iter": 0, "iter_decode_with_beam": 1}' \
#     --eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu \
#     --maximize-best-checkpoint-metric --decoder-learned-pos --encoder-learned-pos \
#     --apply-bert-init --activation-fn gelu --user-dir glat_plugins --max-epoch 10000 \
#     --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp --save-interval-updates 30000 \
#




# CUDA_VISIBLE_DEVICES=7 python3 train.py ${data_dir} \
#     --ddp-backend=legacy_ddp \
#     --task translation_lev \
#     --criterion nat_loss \
#     --save-dir ${save_path} \
#     --arch levenshtein_transformer \
#     --noise random_delete \
#     --share-all-embeddings \
#     --optimizer adam --adam-betas '(0.9,0.98)' \
#     --lr 0.0005 --lr-scheduler inverse_sqrt \
#     --stop-min-lr '1e-09' --warmup-updates 100 \
#     --warmup-init-lr '1e-07' --label-smoothing 0.1 \
#     --dropout 0.3 --weight-decay 0.01 \
#     --decoder-learned-pos \
#     --encoder-learned-pos \
#     --apply-bert-init \
#     --log-format 'simple' --log-interval 30000 \
#     --fixed-validation-seed 7 \
#     --max-tokens 8000 \
#     --save-interval-updates 30000 \
#     --max-update 10000





CUDA_VISIBLE_DEVICES=7 python3 train.py ${data_dir} \
    --save-dir ${save_path} \
    --task translation \
    --arch transformer_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512  \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --skip-invalid-size-inputs-valid-test \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-interval-updates 30000 \
    --max-update 10000 --no-epoch-checkpoints \ 
