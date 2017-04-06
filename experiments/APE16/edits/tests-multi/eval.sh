#!/usr/bin/env bash

main_dir=current/tests-multi

./seq2seq.sh ${main_dir}/global/config.yaml --align current/data/dev.1.{mt,src,edits} --output ${main_dir}/output --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/global.svg
./seq2seq.sh ${main_dir}/global/config.yaml --align current/data/dev.1.{mt,src,edits} --output ${main_dir}/output --align-encoder-id 1 --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/src.global.svg

./seq2seq.sh ${main_dir}/local/config.yaml --align current/data/dev.1.{mt,src,edits} --output ${main_dir}/output --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/local.svg
./seq2seq.sh ${main_dir}/local/config.yaml --align current/data/dev.1.{mt,src,edits} --output ${main_dir}/output --align-encoder-id 1 --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/src.local.svg

./seq2seq.sh ${main_dir}/global_syn/config.yaml --align current/data/dev.1.{mt,src,edits} --output ${main_dir}/output --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/global-syn.svg
./seq2seq.sh ${main_dir}/global_syn/config.yaml --align current/data/dev.1.{mt,src,edits} --output ${main_dir}/output --align-encoder-id 1 --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/src.global-syn.svg

./seq2seq.sh ${main_dir}/local_syn/config.yaml --align current/data/dev.1.{mt,src,edits} --output ${main_dir}/output --align-encoder-id 1 --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/src.local-syn.svg

printf "global     "; ./seq2seq.sh ${main_dir}/global/config.yaml --eval current/data/dev.{mt,src,edits}     2>&1 | tail -n1 | sed "s/.*dev //"
printf "local      ";./seq2seq.sh ${main_dir}/local/config.yaml --eval current/data/dev.{mt,src,edits}       2>&1 | tail -n1 | sed "s/.*dev //"
printf "global-syn ";./seq2seq.sh ${main_dir}/global_syn/config.yaml --eval current/data/dev.{mt,src,edits}  2>&1 | tail -n1 | sed "s/.*dev //"
printf "local-syn  ";./seq2seq.sh ${main_dir}/local_syn/config.yaml --eval current/data/dev.{mt,src,edits}   2>&1 | tail -n1 | sed "s/.*dev //"

printf "global     ";./seq2seq.sh ${main_dir}/global/config.yaml --eval current/data/test.{mt,src,edits}     2>&1 | tail -n1 | sed "s/.*dev //"
printf "local      ";./seq2seq.sh ${main_dir}/local/config.yaml --eval current/data/test.{mt,src,edits}      2>&1 | tail -n1 | sed "s/.*dev //"
printf "global-syn ";./seq2seq.sh ${main_dir}/global_syn/config.yaml --eval current/data/test.{mt,src,edits} 2>&1 | tail -n1 | sed "s/.*dev //"
printf "local-syn  ";./seq2seq.sh ${main_dir}/local_syn/config.yaml --eval current/data/test.{mt,src,edits}  2>&1 | tail -n1 | sed "s/.*dev //"