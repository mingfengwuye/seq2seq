#!/usr/bin/env bash

# Summary of the task at: http://www.statmt.org/wmt16/ape-task.html
# interesting papers on the subject:
# https://arxiv.org/abs/1606.07481:   multi-source, no additional data, predicts a sequence of edits
# https://arxiv.org/abs/1605.04800:   merges two mono-source models, lots of additional (parallel data), creates
# synthetic PE data by using back-translation

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/edits/data_truecase

rm -rf ${data_dir}
mkdir -p ${data_dir}

cat ${raw_data}/train.{mt,pe} > ${data_dir}/concat.de

scripts/train-truecaser.perl --model ${data_dir}/truecaser.de --corpus ${data_dir}/concat.de
scripts/train-truecaser.perl --model ${data_dir}/truecaser.src --corpus ${raw_data}/train.src

scripts/truecase.perl --model ${data_dir}/truecaser.src < ${raw_data}/train.src > ${data_dir}/train.src 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.src < ${raw_data}/dev.src > ${data_dir}/dev.src 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.src < ${raw_data}/test.src > ${data_dir}/test.src 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.src < ${raw_data}/500K.src > ${data_dir}/500K.src 2>/dev/null
0scripts/truecase.perl --model ${data_dir}/truecaser.src < ${raw_data}/4M.src > ${data_dir}/4M.src 2>/dev/null

scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/train.mt > ${data_dir}/train.mt 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/dev.mt > ${data_dir}/dev.mt 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/test.mt > ${data_dir}/test.mt 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/500K.mt > ${data_dir}/500K.mt 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/4M.mt > ${data_dir}/4M.mt 2>/dev/null

scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/train.pe > ${data_dir}/train.pe 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/dev.pe > ${data_dir}/dev.pe 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/test.pe > ${data_dir}/test.pe 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/500K.pe > ${data_dir}/500K.pe 2>/dev/null
scripts/truecase.perl --model ${data_dir}/truecaser.de < ${raw_data}/4M.pe > ${data_dir}/4M.pe 2>/dev/null

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
