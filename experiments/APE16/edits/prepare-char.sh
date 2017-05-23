#!/usr/bin/env bash

#!/usr/bin/env bash

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/edits/data_char

rm -rf ${data_dir}
mkdir -p ${data_dir}

cp ${raw_data}/{train,dev,test,500K}.{src,mt,pe} ${data_dir}

scripts/extract-edits.py ${data_dir}/train.{mt,pe} > ${data_dir}/train.edits --char-level --subs
scripts/extract-edits.py ${data_dir}/dev.{mt,pe} > ${data_dir}/dev.edits --char-level --subs
scripts/extract-edits.py ${data_dir}/test.{mt,pe} > ${data_dir}/test.edits --char-level --subs
scripts/extract-edits.py ${data_dir}/500K.{mt,pe} > ${data_dir}/500K.edits --char-level --subs

scripts/prepare-data.py ${data_dir}/train src pe mt edits ${data_dir} --mode vocab --vocab-size 0 --character-level src pe mt


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
--vocab-size 0 --character-level src pe mt
