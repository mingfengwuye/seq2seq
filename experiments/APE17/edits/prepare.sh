#!/usr/bin/env bash

raw_data=experiments/APE17/raw_data
data_dir=experiments/APE17/edits/data

rm -rf ${data_dir}
mkdir -p ${data_dir}

scripts/extract-edits.py ${raw_data}/train.{mt,pe} > ${raw_data}/train.edits
scripts/extract-edits.py ${raw_data}/dev.{mt,pe} > ${raw_data}/dev.edits

scripts/prepare-data.py ${raw_data}/train src pe mt edits ${data_dir} --no-tokenize \
--dev-size 2000 --test-corpus ${raw_data}/dev --dev-prefix train-dev --test-prefix dev

### more training data
#
#data_dir=experiments/APE17/edits/data_plus
#
#rm -rf ${data_dir}
#mkdir -p ${data_dir}
#
#cp ${raw_data}/500K.mt ${data_dir}/concat.mt
#cp ${raw_data}/500K.pe ${data_dir}/concat.pe
#cp ${raw_data}/500K.src ${data_dir}/concat.src
#cp ${raw_data}/500K.edits ${data_dir}/concat.edits
#
#for i in {1..10}; do   # oversample PE data
#    cat ${raw_data}/train.mt >> ${data_dir}/concat.mt
#    cat ${raw_data}/train.pe >> ${data_dir}/concat.pe
#    cat ${raw_data}/train.src >> ${data_dir}/concat.src
#    cat ${raw_data}/train.edits >> ${data_dir}/concat.edits
#done
#
#scripts/prepare-data.py ${data_dir}/concat src pe mt edits ${data_dir} --no-tokenize \
#--dev-corpus ${raw_data}/dev --test-corpus ${raw_data}/test --shuffle
