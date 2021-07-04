# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import math
import pickle

def run(decoder_embed_dim,decoder_layers,decoder_attention_heads):
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
              --max-update 10"
    print(search_command)
    os.system(search_command)

    perplexity_score = float(open('.perplexity.log','r').read().strip())
    return perplexity_score

def postprocess(field):
    params = []
    for i in field.split(', ')[1:]:
        params.append(i.split(': ')[-1])
    return params

with open(f'{sys.argv[1]}.pkl', 'rb') as handle:
    params_list = pickle.load(handle)

# run all
result_list = []
for tups in params_list:
    decoder_embed_dim,decoder_layers,decoder_attention_heads = postprocess(tups[0][2])
    perplexity_score = run(decoder_embed_dim,decoder_layers,decoder_attention_heads)
    tups.append(perplexity_score)
    result_list.append(tups)
