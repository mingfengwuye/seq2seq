#!/usr/bin/env bash

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/edits/data

#rm -rf ${data_dir}
mkdir -p ${data_dir}

scripts/extract-edits.py ${raw_data}/train.{mt,pe} > ${data_dir}/train.char.edits --char-level
scripts/extract-edits.py ${raw_data}/dev.{mt,pe} > ${data_dir}/dev.char.edits --char-level
scripts/extract-edits.py ${raw_data}/test.{mt,pe} > ${data_dir}/test.char.edits --char-level

cat ${raw_data}/train.{src,mt,pe} > ${data_dir}/train.char.all

cp ${raw_data}/train.src ${data_dir}/train.char.src
cp ${raw_data}/train.mt ${data_dir}/train.char.mt
cp ${raw_data}/train.pe ${data_dir}/train.char.pe
cp ${raw_data}/dev.src ${data_dir}/dev.char.src
cp ${raw_data}/dev.mt ${data_dir}/dev.char.mt
cp ${raw_data}/dev.pe ${data_dir}/dev.char.pe
cp ${raw_data}/test.src ${data_dir}/test.char.src
cp ${raw_data}/test.mt ${data_dir}/test.char.mt
cp ${raw_data}/test.pe ${data_dir}/test.char.pe

scripts/prepare-data.py ${data_dir}/train.char all ${data_dir} --mode vocab --vocab-size 0 --character-level --vocab-prefix vocab.char

cp ${data_dir}/vocab.char.all ${data_dir}/vocab.char.mt
cp ${data_dir}/vocab.char.all ${data_dir}/vocab.char.pe
cp ${data_dir}/vocab.char.all ${data_dir}/vocab.char.edits
cp ${data_dir}/vocab.char.all ${data_dir}/vocab.char.src
