#!/bin/bash


for l in fr de it es ru nl ja cs zh tr fi lt
do
  rm $l/*spm*
  rm -rf $l/en_$l
  rm -rf $l/ib_$l
done
