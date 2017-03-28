#!/usr/bin/env bash

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/edits/data_subwords_synthetic

rm -rf ${data_dir}
mkdir -p ${data_dir}

cat ${raw_data}/500K.src > ${data_dir}/concat.src
cat ${raw_data}/500K.mt > ${data_dir}/concat.mt
cat ${raw_data}/500K.pe > ${data_dir}/concat.pe
for i in {1..20}; do   # oversample PE data
    cat ${raw_data}/train.src >> ${data_dir}/concat.src
    cat ${raw_data}/train.mt >> ${data_dir}/concat.mt
    cat ${raw_data}/train.pe >> ${data_dir}/concat.pe
done

cat ${data_dir}/concat.{mt,pe} > ${data_dir}/concat.de

# FIXME: feels like cheating
scripts/learn_bpe.py -s 30000 < ${data_dir}/concat.de > ${data_dir}/bpe.de
scripts/learn_bpe.py -s 30000 < ${data_dir}/concat.src > ${data_dir}/bpe.src

scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${data_dir}/concat.pe > ${data_dir}/train.concat.pe
scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${data_dir}/concat.mt > ${data_dir}/train.concat.mt
scripts/apply_bpe.py -c ${data_dir}/bpe.src < ${data_dir}/concat.src > ${data_dir}/train.concat.src

scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${raw_data}/4M.pe > ${data_dir}/4M.pe
scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${raw_data}/4M.mt > ${data_dir}/4M.mt
scripts/apply_bpe.py -c ${data_dir}/bpe.src < ${raw_data}/4M.src > ${data_dir}/4M.src

scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${raw_data}/train.pe > ${data_dir}/train.pe
scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${raw_data}/train.mt > ${data_dir}/train.mt
scripts/apply_bpe.py -c ${data_dir}/bpe.src < ${raw_data}/train.src > ${data_dir}/train.src

scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${raw_data}/dev.pe > ${data_dir}/dev.pe
scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${raw_data}/dev.mt > ${data_dir}/dev.mt
scripts/apply_bpe.py -c ${data_dir}/bpe.src < ${raw_data}/dev.src > ${data_dir}/dev.src

scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${raw_data}/test.pe > ${data_dir}/test.pe
scripts/apply_bpe.py -c ${data_dir}/bpe.de < ${raw_data}/test.mt > ${data_dir}/test.mt
scripts/apply_bpe.py -c ${data_dir}/bpe.src < ${raw_data}/test.src > ${data_dir}/test.src

scripts/extract-edits.py ${data_dir}/train.concat.{mt,pe} > ${data_dir}/train.concat.edits
scripts/extract-edits.py ${data_dir}/4M.{mt,pe} > ${data_dir}/4M.edits
scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits
scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits

scripts/prepare-data.py ${data_dir}/train src mt edits ${data_dir} --mode vocab --vocab-size 0
scripts/prepare-data.py ${data_dir}/train.concat src mt edits ${data_dir} --mode vocab --vocab-size 0 \
--vocab-prefix vocab.concat