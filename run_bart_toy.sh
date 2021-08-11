#!/bin/bash


python examples/bart/summarize.py   --model-dir .   \
--model-file checkpoints/bart.large.cnn/model.pt    \
--src ~/fairseq_cnn_data/cnn_cln/toy_me.source    \
--out ~/fairseq_cnn_data/cnn_cln/test.hypo;
