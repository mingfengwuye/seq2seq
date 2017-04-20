#!/usr/bin/env bash

main_dir=`pwd`/experiments/APE17/synthetic/SMT
data_dir=`pwd`/experiments/APE17/synthetic/SMT/data
raw_data=`pwd`/experiments/APE17/raw_data
mosesdecoder=~/local/moses

mkdir -p ${data_dir}
cp ${raw_data}/{train,train-dev}.{src,mt,pe} ${data_dir}

${mosesdecoder}/bin/lmplz -o 3 < ${data_dir}/train.src > ${data_dir}/train.src.arpa
${mosesdecoder}/bin/lmplz -o 3 < ${data_dir}/train.mt > ${data_dir}/train.mt.arpa


# Train PE->MT model
model_dir=${main_dir}/PE_MT  # maybe do PE->SRC->MT to add more noise
rm -rf ${model_dir}
mkdir -p ${model_dir}

${mosesdecoder}/scripts/training/train-model.perl -root-dir ${model_dir} \
-corpus ${data_dir}/train -f pe -e mt -alignment grow-diag-final-and \
-reordering msd-bidirectional-fe -lm 0:3:${data_dir}/train.mt.arpa:8 \
-mgiza -external-bin-dir ${mosesdecoder}/training-tools \
-mgiza-cpus 8 -cores 8 --parallel

# filter phrase table
# ${mosesdecoder}/scripts/training/filter-model-given-input.pl ${output_dir}_filtered ${output_dir} ${output_dir}/concat.fr

${mosesdecoder}/scripts/training/mert-moses.pl ${data_dir}/train-dev.pe ${data_dir}/train-dev.mt \
${mosesdecoder}/bin/moses ${model_dir}/model/moses.ini --mertdir ${mosesdecoder}/bin/ \
--decoder-flags="-threads 8" &> ${model_dir}/tuning.log --working-dir ${model_dir}/mert-work
# --no-filter-phrase-table

mv ${model_dir}/mert-work/moses.ini ${model_dir}/moses.tuned.ini
rm -rf ${model_dir}/mert-work

# Train PE->SRC model

model_dir=${main_dir}/PE_SRC
rm -rf ${model_dir}
mkdir -p ${model_dir}

${mosesdecoder}/scripts/training/train-model.perl -root-dir ${model_dir} \
-corpus ${data_dir}/train -f pe -e src -alignment grow-diag-final-and \
-reordering msd-bidirectional-fe -lm 0:3:${data_dir}/train.src.arpa:8 \
-mgiza -external-bin-dir ${mosesdecoder}/training-tools \
-mgiza-cpus 8 -cores 8 --parallel

${mosesdecoder}/scripts/training/mert-moses.pl ${data_dir}/train-dev.pe ${data_dir}/train-dev.src \
${mosesdecoder}/bin/moses ${model_dir}/model/moses.ini --mertdir ${mosesdecoder}/bin/ \
--decoder-flags="-threads 8" &> ${model_dir}/tuning.log --working-dir ${model_dir}/mert-work

mv ${model_dir}/mert-work/moses.ini ${model_dir}/moses.tuned.ini
rm -rf ${model_dir}/mert-work

# Train SRC->MT model

model_dir=${main_dir}/SRC_MT
rm -rf ${model_dir}
mkdir -p ${model_dir}

${mosesdecoder}/scripts/training/train-model.perl -root-dir ${model_dir} \
-corpus ${data_dir}/train -f src -e mt -alignment grow-diag-final-and \
-reordering msd-bidirectional-fe -lm 0:3:${data_dir}/train.mt.arpa:8 \
-mgiza -external-bin-dir ${mosesdecoder}/training-tools \
-mgiza-cpus 8 -cores 8 --parallel

${mosesdecoder}/scripts/training/mert-moses.pl ${data_dir}/train-dev.src ${data_dir}/train-dev.mt \
${mosesdecoder}/bin/moses ${model_dir}/model/moses.ini --mertdir ${mosesdecoder}/bin/ \
--decoder-flags="-threads 8" &> ${model_dir}/tuning.log --working-dir ${model_dir}/mert-work

mv ${model_dir}/mert-work/moses.ini ${model_dir}/moses.tuned.ini
rm -rf ${model_dir}/mert-work
