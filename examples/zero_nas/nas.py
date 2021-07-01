# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os

search_command = f"python train.py --task language_modeling \
          data-bin/wikitext-2 \
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
