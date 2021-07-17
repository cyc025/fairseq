#!/bin/bash


# python examples/bart/summarize.py   --model-dir .   --model-file checkpoints/bart.base/model.pt   --src ~/fairseq_cnn_data/cnn_cln/toy_test.source   --out ~/fairseq_cnn_data/cnn_cln/test.hypo;
python examples/bart/summarize.py   --model-dir .  --model-file checkpoints/bart.large.cnn/model.pt   --src ~/fairseq_cnn_data/cnn_cln/test.source   --out ~/fairseq_cnn_data/cnn_cln/test.hypo;
