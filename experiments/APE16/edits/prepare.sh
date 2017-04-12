#!/usr/bin/env bash

# Summary of the task at: http://www.statmt.org/wmt16/ape-task.html
# interesting papers on the subject:
# https://arxiv.org/abs/1606.07481:   multi-source, no additional data, predicts a sequence of edits
# https://arxiv.org/abs/1605.04800:   merges two mono-source models, lots of additional (parallel data), creates
# synthetic PE data by using back-translation

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/edits/data

rm -rf ${data_dir}
mkdir -p ${data_dir}

cp ${raw_data}/{train,dev,test,500K,4M}.{src,mt,pe} ${data_dir}

scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits
scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits
scripts/extract-edits.py ${data_dir}/500K.{mt,pe} > ${data_dir}/500K.edits
scripts/extract-edits.py ${data_dir}/4M.{mt,pe} > ${data_dir}/4M.edits

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

# train.all dataset
cat ${data_dir}/{4M,train.concat}.mt > ${data_dir}/train.all.mt
cat ${data_dir}/{4M,train.concat}.pe > ${data_dir}/train.all.pe
cat ${data_dir}/{4M,train.concat}.src > ${data_dir}/train.all.src
cat ${data_dir}/{4M,train.concat}.edits > ${data_dir}/train.all.edits

scripts/prepare-data.py ${data_dir}/train.all src pe mt edits ${data_dir} --mode vocab --vocab-prefix vocab.all \
--vocab-size 60000

# smaller vocab
head -n30000 ${data_dir}/vocab.all.mt > ${data_dir}/vocab.all.30k.mt
head -n30000 ${data_dir}/vocab.all.pe > ${data_dir}/vocab.all.30k.pe
head -n30000 ${data_dir}/vocab.all.src > ${data_dir}/vocab.all.30k.src
head -n30000 ${data_dir}/vocab.all.edits > ${data_dir}/vocab.all.30k.edits

# train.100k dataset
head -n100000 ${data_dir}/500K.mt > ${data_dir}/train.100k.mt
head -n100000 ${data_dir}/500K.pe > ${data_dir}/train.100k.pe
head -n100000 ${data_dir}/500K.src > ${data_dir}/train.100k.src
head -n100000 ${data_dir}/500K.edits > ${data_dir}/train.100k.edits
cat ${data_dir}/train.mt >> ${data_dir}/train.100k.mt
cat ${data_dir}/train.pe >> ${data_dir}/train.100k.pe
cat ${data_dir}/train.src >> ${data_dir}/train.100k.src
cat ${data_dir}/train.edits >> ${data_dir}/train.100k.edits

scripts/prepare-data.py ${data_dir}/train.100k src pe mt edits ${data_dir} --mode vocab --vocab-prefix vocab.100k \
--vocab-size 30000
