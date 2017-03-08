#!/usr/bin/env bash

# raw_data=experiments/APE17/raw_data
# data_dir=experiments/APE17/data_edits
# rm -rf ${data_dir}
# mkdir -p ${data_dir}
# scripts/extract-edits.py ${raw_data}/train.{mt,pe} > ${data_dir}/train.edits
# scripts/extract-edits.py ${raw_data}/dev.{mt,pe} > ${data_dir}/dev.edits

# scripts/prepare-data.py ${raw_data}/train src pe mt edits ${data_dir} \
# --no-tokenize --dev-size 2000 --dev-prefix train-dev --test-prefix dev --test-corpus ${raw_data}/dev

raw_data=experiments/APE16/raw_data
data_dir=experiments/APE16/synthetic/data
mosesdecoder=~/local/moses

rm -rf ${data_dir}
mkdir -p ${data_dir}

# 1. train truecase model
# 2. apply truecasing
# 3. train subwords

cat ${raw_data}/train.src > ${data_dir}/raw.src
cat ${raw_data}/train.mt > ${data_dir}/raw.de
cat ${raw_data}/train.pe >> ${data_dir}/raw.de

${mosesdecoder}/scripts/recaser/train-truecaser.perl --model ${data_dir}/truecaser.src --corpus ${data_dir}/raw.src
${mosesdecoder}/scripts/recaser/train-truecaser.perl --model ${data_dir}/truecaser.de --corpus ${data_dir}/raw.de

${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.src < ${data_dir}/raw.src > ${data_dir}/raw.true.src 2>/dev/null
${mosesdecoder}/scripts/recaser/truecase.perl --model ${data_dir}/truecaser.de < ${data_dir}/raw.de > ${data_dir}/raw.true.de 2>/dev/null

scripts/prepare-data.py ${data_dir}/raw.true src de ${data_dir} --subwords --vocab-size 40000 --no-tokenize --output trash
rm -f ${data_dir}/{raw,raw.true,trash}.{src,mt,de}
cp ${data_dir}/bpe.de ${data_dir}/bpe.mt
cp ${data_dir}/bpe.de ${data_dir}/bpe.pe
