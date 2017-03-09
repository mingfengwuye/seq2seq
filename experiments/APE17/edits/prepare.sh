#!/usr/bin/env bash

raw_data=experiments/APE17/raw_data
data_dir=experiments/APE17/edits/data

rm -rf ${data_dir}
mkdir -p ${data_dir}

scripts/extract-edits.py ${raw_data}/train.{mt,pe} > ${raw_data}/train.edits
scripts/extract-edits.py ${raw_data}/dev.{mt,pe} > ${raw_data}/dev.edits

scripts/prepare-data.py ${raw_data}/train src pe mt edits ${data_dir} --no-tokenize \
--dev-size 2000 --test-corpus ${raw_data}/dev --dev-prefix train-dev --test-prefix dev
