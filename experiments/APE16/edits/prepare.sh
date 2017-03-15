#!/usr/bin/env bash

# Summary of the task at: http://www.statmt.org/wmt16/ape-task.html
# interesting papers on the subject:
# https://arxiv.org/abs/1606.07481:   multi-source, no additional data, predicts a sequence of edits
# https://arxiv.org/abs/1605.04800:   merges two mono-source models, lots of additional (parallel data), creates
# synthetic PE data by using back-translation

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/edits/data

if [ ! -f ${raw_data}/train.edits ]; then
    scripts/extract-edits.py ${raw_data}/train.{mt,pe} > ${raw_data}/train.edits
fi
if [ ! -f ${raw_data}/dev.edits ]; then
    scripts/extract-edits.py ${raw_data}/dev.{mt,pe} > ${raw_data}/dev.edits
fi
if [ ! -f ${raw_data}/test.edits ]; then
    scripts/extract-edits.py ${raw_data}/test.{mt,pe} > ${raw_data}/test.edits
fi
if [ ! -f ${raw_data}/500K.edits ]; then
    scripts/extract-edits.py ${raw_data}/500K.{mt,pe} > ${raw_data}/500K.edits
fi

rm -rf ${data_dir}
mkdir -p ${data_dir}

scripts/prepare-data.py ${raw_data}/train src pe mt edits ${data_dir} --no-tokenize \
--dev-corpus ${raw_data}/dev \
--test-corpus ${raw_data}/test

cp ${raw_data}/500K.mt ${data_dir}/concat.mt
cp ${raw_data}/500K.pe ${data_dir}/concat.pe
cp ${raw_data}/500K.src ${data_dir}/concat.src
cp ${raw_data}/500K.edits ${data_dir}/concat.edits

for i in {1..20}; do   # oversample PE data
    cat ${raw_data}/train.mt >> ${data_dir}/concat.mt
    cat ${raw_data}/train.pe >> ${data_dir}/concat.pe
    cat ${raw_data}/train.src >> ${data_dir}/concat.src
    cat ${raw_data}/train.edits >> ${data_dir}/concat.edits
done

scripts/prepare-data.py ${data_dir}/concat src pe mt edits ${data_dir} --no-tokenize --shuffle --output train-concat \
--vocab-prefix vocab-concat --vocab-size 30000

# TODO: truecase data