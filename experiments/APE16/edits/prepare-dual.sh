#!/usr/bin/env bash

# Summary of the task at: http://www.statmt.org/wmt16/ape-task.html
# interesting papers on the subject:
# https://arxiv.org/abs/1606.07481:   multi-source, no additional data, predicts a sequence of edits
# https://arxiv.org/abs/1605.04800:   merges two mono-source models, lots of additional (parallel data), creates
# synthetic PE data by using back-translation

raw_data=experiments/APE16/raw_data

subs=false

if [ "$subs" = true ] ; then
    data_dir=experiments/APE16/edits/data_dual_sub
else
    data_dir=experiments/APE16/edits/data_dual
fi

rm -rf ${data_dir}
mkdir -p ${data_dir}

cp ${raw_data}/{train,dev,test}.{src,mt,pe} ${data_dir}

if [ "$subs" = true ] ; then
    scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits.ops --subs --ops-only
    scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits.ops --subs --ops-only
    scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits.ops --subs --ops-only

    scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits.words --subs --words-only
    scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits.words --subs --words-only
    scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits.words --subs --words-only
else
    scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits.ops --ops-only
    scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits.ops --ops-only
    scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits.ops --ops-only

    scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits.words --words-only
    scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits.words --words-only
    scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits.words --words-only
fi

cat ${data_dir}/train.{mt,pe} > ${data_dir}/train.de

scripts/prepare-data.py ${data_dir}/train src de edits.ops ${data_dir} --mode vocab --vocab-size 30000

cp ${data_dir}/vocab.de ${data_dir}/vocab.mt
cp ${data_dir}/vocab.de ${data_dir}/vocab.pe
cp ${data_dir}/vocab.de ${data_dir}/vocab.edits.words

cp ${raw_data}/500K.mt ${data_dir}/train.concat.mt
cp ${raw_data}/500K.src ${data_dir}/train.concat.src
cp ${raw_data}/500K.pe ${data_dir}/train.concat.pe

if [ "$subs" = true ] ; then
    scripts/extract-edits.py ${data_dir}/train.concat.{mt,pe} > ${data_dir}/train.concat.edits.ops --subs --ops-only
    scripts/extract-edits.py ${data_dir}/train.concat.{mt,pe} > ${data_dir}/train.concat.edits.words --subs --words-only
else
    scripts/extract-edits.py ${data_dir}/train.concat.{mt,pe} > ${data_dir}/train.concat.edits.ops --ops-only
    scripts/extract-edits.py ${data_dir}/train.concat.{mt,pe} > ${data_dir}/train.concat.edits.words --words-only
fi

for i in {1..20}; do   # oversample PE data
    cat ${data_dir}/train.mt >> ${data_dir}/train.concat.mt
    cat ${data_dir}/train.pe >> ${data_dir}/train.concat.pe
    cat ${data_dir}/train.src >> ${data_dir}/train.concat.src
    cat ${data_dir}/train.edits.words >> ${data_dir}/train.concat.edits.words
    cat ${data_dir}/train.edits.ops >> ${data_dir}/train.concat.edits.ops
done

cat ${data_dir}/train.concat.{mt,pe} > ${data_dir}/train.concat.de

scripts/prepare-data.py ${data_dir}/train.concat src de edits.ops ${data_dir} --mode vocab --vocab-size 30000 \
--vocab-prefix vocab.concat

cp ${data_dir}/vocab.concat.de ${data_dir}/vocab.concat.mt
cp ${data_dir}/vocab.concat.de ${data_dir}/vocab.concat.pe
cp ${data_dir}/vocab.concat.de ${data_dir}/vocab.concat.edits.words
