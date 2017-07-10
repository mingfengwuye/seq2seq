#!/usr/bin/env bash

# Summary of the task at: http://www.statmt.org/wmt16/ape-task.html
# interesting papers on the subject:
# https://arxiv.org/abs/1606.07481:   multi-source, no additional data, predicts a sequence of edits
# https://arxiv.org/abs/1605.04800:   merges two mono-source models, lots of additional (parallel data), creates
# synthetic PE data by using back-translation

raw_data=experiments/APE16/raw_data
data_dir=experiments/WMT17/data_2016

rm -rf ${data_dir}
mkdir -p ${data_dir}

cp ${raw_data}/{dev,test,500K,train}.{src,mt,pe} ${data_dir}
cp ${raw_data}/test.2017.{src,mt} ${data_dir}

scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits
scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits
scripts/extract-edits.py ${data_dir}/500K.{mt,pe} > ${data_dir}/500K.edits

scripts/prepare-data.py ${data_dir}/train src pe mt edits ${data_dir} --mode vocab --vocab-size 0

cp ${data_dir}/500K.mt ${data_dir}/train.concat.mt
cp ${data_dir}/500K.pe ${data_dir}/train.concat.pe
cp ${data_dir}/500K.src ${data_dir}/train.concat.src
cp ${data_dir}/500K.edits ${data_dir}/train.concat.edits

for i in {1..20}; do   # oversample PE data
    cat ${data_dir}/train.mt >> ${data_dir}/train.concat.mt
    cat ${data_dir}/train.pe >> ${data_dir}/train.concat.pe
    cat ${data_dir}/train.src >> ${data_dir}/train.concat.src
    cat ${data_dir}/train.edits >> ${data_dir}/train.concat.edits
done

scripts/prepare-data.py ${data_dir}/train.concat src pe mt edits ${data_dir} --mode vocab --vocab-prefix vocab.concat \
--vocab-size 30000

scripts/prepare-data.py ${data_dir}/train src mt pe ${data_dir} --vocab-prefix vocab.subwords --output train.subwords \
--dev-corpus ${raw_data}/dev --dev-prefix dev.subwords --test-corpus ${raw_data}/test --test-prefix test.subwords \
--no-tokenize --subwords --vocab-size 0

scripts/prepare-data.py ${data_dir}/train.concat src mt pe ${data_dir} --vocab-prefix vocab.concat.subwords \
--output train.concat.subwords --dev-corpus ${raw_data}/dev --dev-prefix dev.concat.subwords \
--test-corpus ${raw_data}/test --test-prefix test.concat.subwords \
--no-tokenize --subwords --vocab-size 30000
