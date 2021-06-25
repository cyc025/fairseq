#!/bin/bash

upperlim=2
for ((i=2; i<=upperlim; i++)); do
  echo $i > .max.len
  taskset --cpu-list 1 python examples/bart/summarize.py   --model-dir .   --model-file checkpoints/bart.large.cnn/model.pt   --src ~/fairseq_cnn_data/cnn_cln/toy_test.source   --out ~/fairseq_cnn_data/cnn_cln/test.hypo;
  rm .max.len
done

#
# #!/bin/bash
#
# shopt -s expand_aliases;
#
# # to be deleted
# alias with-proxy='env http_proxy=fwdproxy:8080 https_proxy=fwdproxy:8080 no_proxy=.fbcdn.net,.facebook.com,.thefacebook.com,.tfbnw.net,.fb.com,.fburl.com,.facebook.net,.sb.fbsbx.com,localhost RSYNC_PROXY=fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080 HTTPS_PROXY=http://fwdproxy:8080'
#
# upperlim=100
# for ((i=2; i<=upperlim; i++)); do
#     for ((j=1; j<=10; j++)); do
#         echo $i > .curr_index
#         echo $i > .max.len
#         with-proxy taskset --cpu-list 1 python examples/bart/summarize.py   --model-dir data-bin/cnn_dm --model-file ~/checkpoints/bart.large.cnn/model.pt   --src data-bin/cnn_dm/toy_test.source   --out data-bin/cnn_dm/test.hypo;
#         rm .max.len .curr_index
#     done
# done
#
# # upperlim=2
# # for ((i=2; i<=upperlim; i++)); do
# #     echo $i > .curr_index
# #     echo $i > .max.len
# #     with-proxy taskset --cpu-list 1 python examples/bart/summarize.py   --model-dir data-bin/cnn_dm --model-file ~/checkpoints/bart.large.cnn/model.pt   --src data-bin/cnn_dm/toy_test.source   --out data-bin/cnn_dm/test.hypo;
# #     rm .max.len .curr_index
# # done
