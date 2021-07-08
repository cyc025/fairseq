# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import math
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


def search(decoder_embed_dim,decoder_layers,decoder_attention_heads):
    # clear checkpoints
    os.system("rm checkpoints/transformer_wikitext-2/*")

    params = {
        'decoder_embed_dim': decoder_embed_dim,
        'decoder_layers': decoder_layers,
        'decoder_attention_heads': decoder_attention_heads,
    }
    parameters = f"--decoder-embed-dim {params['decoder_embed_dim']} \
                   --decoder-layers {params['decoder_layers']} \
                   --decoder-attention-heads {params['decoder_attention_heads']} "

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
              --disable-validation \
              --fp16 \
              --batch-size 1024 \
              {parameters} \
              --max-update 1"
    os.system(search_command)

    zen_score = float(open('.zen_score.log','r').read().strip())
    return zen_score

"""
    Exhaustive search
"""


decoder_embed_dims = range(50,2000,50)
decoder_layerss = range(1,10)
decoder_attention_headss = range(1,10)

# decoder_embed_dims = range(100,200,50)
# decoder_layerss = range(2,3)
# decoder_attention_headss = range(2,3)

zen_scores_tups = []
with open('search.log','w') as search_log:
    for decoder_embed_dim in decoder_embed_dims:
        for decoder_layers in decoder_layerss:
            for decoder_attention_heads in decoder_attention_headss:
                if decoder_embed_dim%decoder_attention_heads!=0: continue
                zen_score = search(decoder_embed_dim,decoder_layers,decoder_attention_heads)
                if math.isinf(zen_score):
                    zen_score = 1000
                num_params = float(open('params.log','r').read())
                zen_scores_tups.append( (zen_score,num_params,f'zen_score: {zen_score}, decoder_embed_dim: {decoder_embed_dim}, decoder_layers: {decoder_layers}, decoder_attention_heads: {decoder_attention_heads}') )

sorted_max_zen_tup = sorted(zen_scores_tups, key=lambda tup: tup[0])
print(sorted_max_zen_tup[-1])

import pickle
with open('expressivity.pkl', 'wb') as handle:
    pickle.dump(sorted_max_zen_tup, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('expressivity.pkl', 'rb') as handle:
#     b = pickle.load(handle)
#
# print(b)

# with open('pp_size.pkl', 'rb') as handle:
#     params_list = pickle.load(handle)
#
# for i in range(0,len(params_list)):
#     tups = params_list[i]
#     print(tups[0][1])
#
#
