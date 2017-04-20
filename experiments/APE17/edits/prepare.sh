#!/usr/bin/env bash

raw_data=experiments/APE17/raw_data
data_dir=experiments/APE17/edits/data

rm -rf ${data_dir}
mkdir -p ${data_dir}

if [ ! -f ${raw_data}/train.edits ]; then
    scripts/extract-edits.py ${raw_data}/train.{mt,pe} > ${raw_data}/train.edits
fi
if [ ! -f ${raw_data}/dev.edits ]; then
    scripts/extract-edits.py ${raw_data}/dev.{mt,pe} > ${raw_data}/dev.edits
fi

scripts/prepare-data.py ${raw_data}/train src pe mt edits ${data_dir} --no-tokenize \
--dev-size 1000 --test-corpus ${raw_data}/dev --dev-prefix train-dev --test-prefix dev
