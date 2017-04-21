#!/usr/bin/env bash

main_dir=`pwd`/experiments/APE17/SMT
data_dir=`pwd`/experiments/APE17/SMT/data
raw_data=`pwd`/experiments/APE17/raw_data
mosesdecoder=~/local/moses

mkdir -p ${data_dir}
cp ${raw_data}/{train,train-dev}.{src,mt,pe} ${data_dir}

${mosesdecoder}/bin/lmplz -o 3 < ${data_dir}/train.pe > ${data_dir}/train.pe.arpa

# Train MT->PE model
model_dir=${main_dir}/MT_PE
rm -rf ${model_dir}
mkdir -p ${model_dir}

${mosesdecoder}/scripts/training/train-model.perl -root-dir ${model_dir} \
-corpus ${data_dir}/train -f mt -e pe -alignment grow-diag-final-and \
-reordering msd-bidirectional-fe -lm 0:3:${data_dir}/train.pe.arpa:8 \
-mgiza -external-bin-dir ${mosesdecoder}/training-tools \
-mgiza-cpus 8 -cores 8 --parallel

${mosesdecoder}/scripts/training/mert-moses.pl ${data_dir}/train-dev.mt ${data_dir}/train-dev.pe \
${mosesdecoder}/bin/moses ${model_dir}/model/moses.ini --mertdir ${mosesdecoder}/bin/ \
--decoder-flags="-threads 8" &> ${model_dir}/tuning.log --working-dir ${model_dir}/mert-work

mv ${model_dir}/mert-work/moses.ini ${model_dir}/moses.tuned.ini
rm -rf ${model_dir}/mert-work
