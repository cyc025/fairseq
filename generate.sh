#!/bin/bash


# data_dir=data_bin
data_dir=~/fairseq/data-bin/iwslt14.tokenized.de-en
save_path=saved_models
src=de
tgt=en

python3 fairseq_cli/generate.py ${data_dir} --path ${save_path}/checkpoint_best.pt \
    --remove-bpe --max-sentences 20 --source-lang ${src} --target-lang ${tgt} \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test \
    --skip-invalid-size-inputs-valid-test --profile --num-workers	32 --distributed-world-size	4 \
    --results-path data_bin	\

#
# fairseq-generate \
#     $data_dir --num-workers	32 \
#     --path ${save_path}/checkpoint_best.pt \
#     --beam 5 --remove-bpe --cpu	--batch-size 128



# --quiet
