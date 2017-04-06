#!/usr/bin/env bash

main_dir=current/tests-chained

./seq2seq.sh ${main_dir}/chained_global/config.yaml --align current/data/dev.1.{mt,src,edits} --output ${main_dir}/output --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/chained-global.svg

./seq2seq.sh ${main_dir}/chained_global_syn/config.yaml --align current/data/dev.1.{mt,src,edits} --output ${main_dir}/output --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/chained-global-syn.svg

./seq2seq.sh ${main_dir}/model.src.yaml --model-dir ${main_dir}/chained --align current/data/dev.1.{src,mt} --output ${main_dir}/output --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/src.chained.svg

./seq2seq.sh ${main_dir}/model.src.yaml --model-dir ${main_dir}/chained_syn --align current/data/dev.1.{src,mt} --output ${main_dir}/output --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/src.chained-syn.svg

./seq2seq.sh ${main_dir}/model.src.yaml --model-dir ${main_dir}/chained_syn_new --align current/data/dev.1.{src,mt} --output ${main_dir}/output --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/src.chained-syn-new.svg

./seq2seq.sh ${main_dir}/model.src.yaml --model-dir ${main_dir}/chained_global --align current/data/dev.1.{src,mt} --output ${main_dir}/output --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/src.chained-global.svg

./seq2seq.sh ${main_dir}/model.src.yaml --model-dir ${main_dir}/chained_global_syn --align current/data/dev.1.{src,mt} --output ${main_dir}/output --no-gpu 2>/dev/null
mv ${main_dir}/output.1.svg ${main_dir}/src.chained-global-syn.svg

printf "chained            "; ./seq2seq.sh ${main_dir}/chained/config.yaml --eval current/data/dev.{mt,src,edits} 2>&1 | tail -n1 | sed "s/.*dev //"
printf "chained-syn        "; ./seq2seq.sh ${main_dir}/chained_syn/config.yaml --eval current/data/dev.{mt,src,edits} 2>&1 | tail -n1 | sed "s/.*dev //"
printf "chained-syn-new    "; ./seq2seq.sh ${main_dir}/chained_syn_new/config.yaml --eval current/data/dev.{mt,src,edits} 2>&1 | tail -n1 | sed "s/.*dev //"
printf "chained-global     "; ./seq2seq.sh ${main_dir}/chained_global/config.yaml --eval current/data/dev.{mt,src,edits} 2>&1 | tail -n1 | sed "s/.*dev //"
printf "chained-global-syn "; ./seq2seq.sh ${main_dir}/chained_global_syn/config.yaml --eval current/data/dev.{mt,src,edits} 2>&1 | tail -n1 | sed "s/.*dev //"

printf "chained            "; ./seq2seq.sh ${main_dir}/chained/config.yaml --eval current/data/test.{mt,src,edits} 2>&1 | tail -n1 | sed "s/.*dev //"
printf "chained-syn        "; ./seq2seq.sh ${main_dir}/chained_syn/config.yaml --eval current/data/test.{mt,src,edits} 2>&1 | tail -n1 | sed "s/.*dev //"
printf "chained-syn-new    "; ./seq2seq.sh ${main_dir}/chained_syn_new/config.yaml --eval current/data/test.{mt,src,edits} 2>&1 | tail -n1 | sed "s/.*dev //"
printf "chained-global     "; ./seq2seq.sh ${main_dir}/chained_global/config.yaml --eval current/data/test.{mt,src,edits} 2>&1 | tail -n1 | sed "s/.*dev //"
printf "chained-global-syn "; ./seq2seq.sh ${main_dir}/chained_global_syn/config.yaml --eval current/data/test.{mt,src,edits} 2>&1 | tail -n1 | sed "s/.*dev //"
