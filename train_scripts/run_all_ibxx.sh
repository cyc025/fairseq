#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


#for l in fr #7
for l in fi #6
#for l in it #5
#for l in cs ja #4
#for l in ru nl #3
#for l in es zh #2
#for l in de #1
#for l in lt tr #0
do
  bash run_ibxx.sh enib $l
done
