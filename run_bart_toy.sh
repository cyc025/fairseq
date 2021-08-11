#!/bin/bash


ROOT=~/fairseq


python examples/bart/summarize.py   --model-dir ${ROOT}/.   \
--model-file ${ROOT}/checkpoints/bart.large.cnn/model.pt    \
--src ~/fairseq_cnn_data/cnn_cln/toy_me.source    \
--out ~/fairseq_cnn_data/cnn_cln/test.hypo;
