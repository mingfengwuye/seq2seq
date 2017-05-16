#!/usr/bin/env bash

# Summary of the task at: http://www.statmt.org/wmt16/ape-task.html
# interesting papers on the subject:
# https://arxiv.org/abs/1606.07481:   multi-source, no additional data, predicts a sequence of edits
# https://arxiv.org/abs/1605.04800:   merges two mono-source models, lots of additional (parallel data), creates
# synthetic PE data by using back-translation

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/edits/data_new

rm -rf ${data_dir}
mkdir -p ${data_dir}

cp ${raw_data}/{train,dev,test}.{src,mt,pe} ${data_dir}

scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits #--subs
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits #--subs
scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits #--subs

cat ${data_dir}/train.{mt,pe} > ${data_dir}/train.de

scripts/prepare-data.py ${data_dir}/train src de ${data_dir} --mode vocab --vocab-size 0

cp ${data_dir}/vocab.de ${data_dir}/vocab.mt
cp ${data_dir}/vocab.de ${data_dir}/vocab.pe
cp ${data_dir}/vocab.de ${data_dir}/vocab.edits
