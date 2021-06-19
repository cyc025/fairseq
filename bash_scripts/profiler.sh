#!/bin/bash

upperlim=17
for ((i=16; i<=upperlim; i++)); do
  echo i > .max.len
  taskset --cpu-list 1 python examples/bart/summarize.py   --model-dir .   --model-file checkpoints/bart.large.cnn/model.pt   --src ~/fairseq_cnn_data/cnn_cln/toy_test.source   --out ~/fairseq_cnn_data/cnn_cln/test.hypo;
  rm .max.len
done
