#!/usr/bin/env bash

# predict sequences of edits with a SUB op
# INS and SUB ops have different embeddings for the target word

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/edits/data_SUB_basic

rm -rf ${data_dir}
mkdir -p ${data_dir}

cp ${raw_data}/{train,dev,test}.{src,mt,pe} ${data_dir}

scripts/extract-edits.py ${data_dir}/train.{mt,pe} --subs > ${data_dir}/train.edits
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} --subs > ${data_dir}/dev.edits
scripts/extract-edits.py ${data_dir}/test.{mt,pe} --subs > ${data_dir}/test.edits

scripts/prepare-data.py ${data_dir}/train mt src edits ${data_dir} --no-tokenize --mode vocab --vocab-size 0
