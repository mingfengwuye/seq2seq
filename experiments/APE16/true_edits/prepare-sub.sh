#!/usr/bin/env bash

# Summary of the task at: http://www.statmt.org/wmt16/ape-task.html
# interesting papers on the subject:
# https://arxiv.org/abs/1606.07481:   multi-source, no additional data, predicts a sequence of edits
# https://arxiv.org/abs/1605.04800:   merges two mono-source models, lots of additional (parallel data), creates
# synthetic PE data by using back-translation

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/true_edits/data_sub

rm -rf ${data_dir}
mkdir -p ${data_dir}

cp ${raw_data}/train.mt ${data_dir}/train.raw.mt
cp ${raw_data}/train.pe ${data_dir}/train.raw.pe
cp ${raw_data}/train.src ${data_dir}/train.raw.src

cp ${raw_data}/dev.mt ${data_dir}/dev.raw.mt
cp ${raw_data}/dev.pe ${data_dir}/dev.raw.pe
cp ${raw_data}/dev.src ${data_dir}/dev.raw.src

cp ${raw_data}/test.mt ${data_dir}/test.raw.mt
cp ${raw_data}/test.pe ${data_dir}/test.raw.pe
cp ${raw_data}/test.src ${data_dir}/test.raw.src

scripts/extract-edits.py ${data_dir}/train.raw.{mt,pe} --subs --ops-only > ${data_dir}/train.raw.edits.ops
scripts/extract-edits.py ${data_dir}/dev.raw.{mt,pe} --subs --ops-only > ${data_dir}/dev.raw.edits.ops
scripts/extract-edits.py ${data_dir}/test.raw.{mt,pe} --subs --ops-only > ${data_dir}/test.raw.edits.ops

scripts/extract-edits.py ${data_dir}/train.raw.{mt,pe} --subs --word > ${data_dir}/train.raw.edits.words
scripts/extract-edits.py ${data_dir}/dev.raw.{mt,pe} --subs --words > ${data_dir}/dev.raw.edits.words
scripts/extract-edits.py ${data_dir}/test.raw.{mt,pe} --subs --words > ${data_dir}/test.raw.edits.words

scripts/prepare-data.py ${data_dir}/train.raw src pe mt edits.words edits.ops ${data_dir} --no-tokenize \
--vocab-size 30000 --dev-corpus ${data_dir}/dev.raw --test-corpus ${data_dir}/test.raw

rm -rf ${data_dir}/*.raw.*

#cp ${raw_data}/500K.mt ${data_dir}/concat.mt
#cp ${raw_data}/500K.pe ${data_dir}/concat.pe
#cp ${raw_data}/500K.src ${data_dir}/concat.src
#cp ${raw_data}/500K.edits ${data_dir}/concat.edits
#
#for i in {1..20}; do   # oversample PE data
#    cat ${raw_data}/train.mt >> ${data_dir}/concat.mt
#    cat ${raw_data}/train.pe >> ${data_dir}/concat.pe
#    cat ${raw_data}/train.src >> ${data_dir}/concat.src
#    cat ${raw_data}/train.edits >> ${data_dir}/concat.edits
#done

#scripts/prepare-data.py ${data_dir}/concat src pe mt edits ${data_dir} --no-tokenize --shuffle --output train-concat \
#--vocab-prefix vocab-concat --vocab-size 30000

#scripts/prepare-data.py ${data_dir}/train src pe mt edits ${data_dir} --no-tokenize --shuffle --output train-concat \
#--vocab-prefix vocab-concat --vocab-size 30000