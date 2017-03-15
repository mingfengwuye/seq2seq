#!/usr/bin/env bash

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/true_edits/data_subwords

rm -rf ${data_dir}
mkdir -p ${data_dir}

cat ${raw_data}/500K.mt > ${data_dir}/concat.mt
cat ${raw_data}/500K.pe > ${data_dir}/concat.pe
for i in {1..20}; do   # oversample PE data
    cat ${raw_data}/train.mt >> ${data_dir}/concat.mt
    cat ${raw_data}/train.pe >> ${data_dir}/concat.pe
done

cat ${data_dir}/concat.{mt,pe} > ${data_dir}/concat.de

scripts/learn_bpe.py -s 40000 < ${data_dir}/concat.de > ${data_dir}/bpe.de
scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${data_dir}/concat.pe > ${data_dir}/train.pe
scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${data_dir}/concat.mt > ${data_dir}/train.mt

scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${raw_data}/dev.pe > ${data_dir}/dev.pe
scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${raw_data}/dev.mt > ${data_dir}/dev.mt

scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${raw_data}/test.pe > ${data_dir}/test.pe
scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${raw_data}/test.mt > ${data_dir}/test.mt

scripts/extract-edits.py ${data_dir}/train.{mt,pe} --ops-only > ${data_dir}/train.edits.ops
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} --ops-only > ${data_dir}/dev.edits.ops
scripts/extract-edits.py ${data_dir}/test.{mt,pe} --ops-only > ${data_dir}/test.edits.ops

scripts/extract-edits.py ${data_dir}/train.{mt,pe} --word > ${data_dir}/train.edits.words
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} --words > ${data_dir}/dev.edits.words
scripts/extract-edits.py ${data_dir}/test.{mt,pe} --words > ${data_dir}/test.edits.words

scripts/prepare-data.py ${data_dir}/train mt edits.ops edits.words ${data_dir} --mode vocab --vocab-size 0
