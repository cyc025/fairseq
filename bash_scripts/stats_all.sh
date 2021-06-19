#!/bin/bash


for l in fr de it es ru nl ja cs zh tr fi lt
do
  wc -l $l/train.en >> stats
  wc -l $l/train.$l >> stats
  wc -l $l/train.ib >> stats
  wc -l $l/test.en >> stats
  wc -l $l/test.$l >> stats
  wc -l $l/test.ib >> stats
  wc -l $l/valid.en >> stats
  wc -l $l/valid.$l >> stats
  wc -l $l/valid.ib >> stats
  rm -rf st_corpus
done
