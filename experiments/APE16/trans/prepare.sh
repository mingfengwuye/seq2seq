#!/usr/bin/env bash

data_dir=experiments/APE16/trans/data
raw_data=experiments/APE16/raw_data
mosesdecoder=~/local/moses

rm -rf ${data_dir}
mkdir -p ${data_dir}

cat ${raw_data}/{4M,500K,train}.mt > ${data_dir}/raw.mt
cat ${raw_data}/{4M,500K,train}.src > ${data_dir}/raw.src
cat ${raw_data}/{4M,500K,train}.pe > ${data_dir}/raw.pe

${mosesdecoder}/scripts/recaser/train-truecaser.perl --model ${data_dir}/truecaser.mt --corpus ${data_dir}/raw.mt
${mosesdecoder}/scripts/recaser/train-truecaser.perl --model ${data_dir}/truecaser.src --corpus ${data_dir}/raw.src
${mosesdecoder}/scripts/recaser/train-truecaser.perl --model ${data_dir}/truecaser.pe --corpus ${data_dir}/raw.pe

${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.mt < ${data_dir}/raw.mt > ${data_dir}/raw.true.mt 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.src < ${data_dir}/raw.src > ${data_dir}/raw.true.src 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.pe < ${data_dir}/raw.pe > ${data_dir}/raw.true.pe 2>/dev/null

scripts/prepare-data.py ${data_dir}/raw.true src mt pe ${data_dir} --subwords --vocab-size 40000 --no-tokenize --output trash
rm ${data_dir}/{raw.true,trash}.{mt,src,pe}

${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.mt < ${raw_data}/4M.mt > ${data_dir}/train.true.mt 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.src < ${raw_data}/4M.src > ${data_dir}/train.true.src 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.pe < ${raw_data}/4M.pe > ${data_dir}/train.true.pe 2>/dev/null

scripts/prepare-data.py ${data_dir}/train.true src mt pe ${data_dir} --mode prepare --output pretrain --no-tokenize --bpe-path ${data_dir}/bpe --subwords
rm ${data_dir}/train.true.{mt,src,pe}

cp ${raw_data}/500K.mt ${data_dir}/raw.mt
cp ${raw_data}/500K.src ${data_dir}/raw.src
cp ${raw_data}/500K.pe ${data_dir}/raw.pe

for i in {1..20}; do   # oversample PE data
    cat ${raw_data}/train.mt >> ${data_dir}/raw.mt
    cat ${raw_data}/train.src >> ${data_dir}/raw.src
    cat ${raw_data}/train.pe >> ${data_dir}/raw.pe
done

# TODO: use same subword units for MT and PE

${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.mt < ${data_dir}/raw.mt > ${data_dir}/train.true.mt 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.src < ${data_dir}/raw.src > ${data_dir}/train.true.src 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.pe < ${data_dir}/raw.pe > ${data_dir}/train.true.pe 2>/dev/null

${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.mt < ${raw_data}/dev.mt > ${data_dir}/dev.true.mt 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.src < ${raw_data}/dev.src > ${data_dir}/dev.true.src 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.pe < ${raw_data}/dev.pe > ${data_dir}/dev.true.pe 2>/dev/null

${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.mt < ${raw_data}/test.mt > ${data_dir}/test.true.mt 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.src < ${raw_data}/test.src > ${data_dir}/test.true.src 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.pe < ${raw_data}/test.pe > ${data_dir}/test.true.pe 2>/dev/null

scripts/prepare-data.py ${data_dir}/train.true src mt pe ${data_dir} --mode prepare --no-tokenize --bpe-path ${data_dir}/bpe --subwords \
--dev-corpus ${data_dir}/dev.true --test-corpus ${data_dir}/test.true
rm ${data_dir}/{train,dev,test}.true.{mt,src,pe}