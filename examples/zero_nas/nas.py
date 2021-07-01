# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

    #
    #
    # [--arch {transformer,transformer_iwslt_de_en,transformer_wmt_en_de,transformer_vaswani_wmt_en_de_big,transformer_vaswani_wmt_en_fr_big,transformer_wmt_en_de_big,transformer_wmt_en_de_big_t2t}]
    # [--activation-fn {relu,gelu,gelu_fast,gelu_accurate,tanh,linear}]
    # [--dropout D] [--attention-dropout D] [--activation-dropout D]
    # [--encoder-embed-path STR] [--encoder-embed-dim N]
    # [--encoder-ffn-embed-dim N] [--encoder-layers N]
    # [--encoder-attention-heads N] [--encoder-normalize-before]
    # [--encoder-learned-pos] [--decoder-embed-path STR]
    # [--decoder-embed-dim N] [--decoder-ffn-embed-dim N]
    # [--decoder-layers N] [--decoder-attention-heads N]
    # [--decoder-learned-pos] [--decoder-normalize-before]
    # [--decoder-output-dim N] [--share-decoder-input-output-embed]
    # [--share-all-embeddings] [--no-token-positional-embeddings]
    # [--adaptive-softmax-cutoff EXPR] [--adaptive-softmax-dropout D]
    # [--layernorm-embedding] [--no-scale-embedding]
    # [--checkpoint-activations] [--no-cross-attention]
    # [--cross-self-attention] [--encoder-layerdrop D]
    # [--decoder-layerdrop D]
    # [--encoder-layers-to-keep ENCODER_LAYERS_TO_KEEP]
    # [--decoder-layers-to-keep DECODER_LAYERS_TO_KEEP] [--quant-noise-pq D]
    # [--quant-noise-pq-block-size D] [--quant-noise-scalar D]


# clear checkpoints
os.system("rm checkpoints/transformer_wikitext-2/*") 

decoder_embed_dim = 200
decoder_layers = 8
decoder_attention_heads = 2
parameters = f"--decoder-embed-dim {decoder_embed_dim} \
               --decoder-layers {decoder_layers} \
               --decoder-attention-heads {decoder_attention_heads} "

search_command = f"python train.py --task language_modeling \
          data-bin/wikitext-2 \
          --decoder-normalize-before \
          --save-dir checkpoints/transformer_wikitext-2 \
          --arch transformer_lm --share-decoder-input-output-embed \
          --dropout 0.1 \
          --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
          --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
          --tokens-per-sample 512 --sample-break-mode none \
          --max-tokens 2048 --update-freq 16 \
          --fp16 \
          --max-update 1"
os.system(search_command)

zen_score = float(open('.zen_score.log','r').read().strip())
print(zen_score)
