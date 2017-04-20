#!/usr/bin/env bash

main_dir=`pwd`/experiments/APE17/synthetic/SMT
data_dir=${main_dir}/data
corpus=`pwd`/experiments/APE17/raw_data/mono.en
split_dir=${main_dir}/split_new

rm -rf ${split_dir}
mkdir -p ${split_dir}

total=`wc -l ${corpus} | cut -d' ' -f1,1`

l=$(($total / 160))

split -l${l} -d -a3 ${corpus} ${split_dir}/mono. --additional-suffix .en
