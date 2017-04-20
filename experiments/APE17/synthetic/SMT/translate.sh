#!/usr/bin/env bash

main_dir=`pwd`/experiments/APE17/synthetic/SMT
split_dir=${main_dir}/split_new
model_dir=${main_dir}/PE_SRC
mosesdecoder=~/tools/moses
#mosesdecoder=~/local/moses

n=$(($# / 2))

for ((i=1; i<=$n; i++))
do
    j=$((i+n))

    in_file=${split_dir}/${!i}
    out_file=${split_dir}/${!j}

    echo ${in_file}" => "${out_file}

    nohup cat ${in_file} | sed "s/|//g" | ${mosesdecoder}/bin/moses -f ${model_dir}/moses.tuned.ini -threads 1 > ${out_file} 2>/dev/null &
done
