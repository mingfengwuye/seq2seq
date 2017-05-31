#!/usr/bin/env bash

# Summary of the task at: http://www.statmt.org/wmt16/ape-task.html
# interesting papers on the subject:
# https://arxiv.org/abs/1606.07481:   multi-source, no additional data, predicts a sequence of edits
# https://arxiv.org/abs/1605.04800:   merges two mono-source models, lots of additional (parallel data), creates
# synthetic PE data by using back-translation

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/edits/data_dual
#data_dir=experiments/APE16/edits/data_dual_nosub

rm -rf ${data_dir}
mkdir -p ${data_dir}

cp ${raw_data}/{train,dev,test}.{src,mt,pe} ${data_dir}

cat ${raw_data}/{train,500K}.mt > ${data_dir}/train.concat.mt
cat ${raw_data}/{train,500K}.src > ${data_dir}/train.concat.src
cat ${raw_data}/{train,500K}.pe > ${data_dir}/train.concat.pe

scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits.ops --subs --ops-only
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits.ops --subs --ops-only
scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits.ops --subs --ops-only

scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits.words --subs --words-only
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits.words --subs --words-only
scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits.words --subs --words-only

scripts/extract-edits.py ${data_dir}/train.concat.{mt,pe} > ${data_dir}/train.concat.edits.ops --subs --ops-only
scripts/extract-edits.py ${data_dir}/train.concat.{mt,pe} > ${data_dir}/train.concat.edits.words --subs --words-only

#scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits.ops --ops-only
#scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits.ops --ops-only
#scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits.ops --ops-only
#
#scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits.words --words-only
#scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits.words --words-only
#scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits.words --words-only
#
#scripts/extract-edits.py ${data_dir}/train.concat.{mt,pe} > ${data_dir}/train.concat.edits.ops --ops-only
#scripts/extract-edits.py ${data_dir}/train.concat.{mt,pe} > ${data_dir}/train.concat.edits.words --words-only

cat ${data_dir}/train.{mt,pe} > ${data_dir}/train.de
cat ${data_dir}/train.concat.{mt,pe} > ${data_dir}/train.concat.de

scripts/prepare-data.py ${data_dir}/train src de edits.ops ${data_dir} --mode vocab --vocab-size 30000

cp ${data_dir}/vocab.de ${data_dir}/vocab.mt
cp ${data_dir}/vocab.de ${data_dir}/vocab.pe
cp ${data_dir}/vocab.de ${data_dir}/vocab.edits.words

scripts/prepare-data.py ${data_dir}/train.concat src de edits.ops ${data_dir} --mode vocab --vocab-size 30000 \
--vocab-prefix vocab.concat

cp ${data_dir}/vocab.concat.de ${data_dir}/vocab.concat.mt
cp ${data_dir}/vocab.concat.de ${data_dir}/vocab.concat.pe
cp ${data_dir}/vocab.concat.de ${data_dir}/vocab.concat.edits.words
