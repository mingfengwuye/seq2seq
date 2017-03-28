#!/usr/bin/env bash

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/edits/data_shared

rm -rf ${data_dir}
mkdir -p ${data_dir}

cp ${raw_data}/{train,dev,test}.{src,mt,pe} ${data_dir}

scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits
scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits

cat ${data_dir}/train.{mt,pe} > ${data_dir}/train.de

scripts/prepare-data.py ${data_dir}/train de ${data_dir} --no-tokenize --mode vocab --vocab-size 0
scripts/prepare-data.py ${data_dir}/train src ${data_dir} --no-tokenize --mode vocab --vocab-size 0

cp ${data_dir}/vocab.de ${data_dir}/vocab.mt
cp ${data_dir}/vocab.de ${data_dir}/vocab.pe
cp ${data_dir}/vocab.de ${data_dir}/vocab.edits
