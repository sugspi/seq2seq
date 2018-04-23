#!/bin/bash

# USAGE: ./extract.sh amf.txt 

file=$1
fname=${file/.txt/}

mkdir -p en_plain_${fname}/${fname}_dec2ans
mkdir -p en_plain_${fname}/${fname}_ans2dec

d2a_dir="en_plain_${fname}/${fname}_dec2ans"
a2d_dir="en_plain_${fname}/${fname}_ans2dec"

cat $file \
  | sed 's/^\s*$/#/g' | tr -d '\n' \
  | sed 's/Answer sentence:/:Answer sentence:/g' \
  | tr '#' '\n' \
  | awk -F':' '{print $2"#"$4}' \
  | nl -n ln -s '#' \
  | sed 's/\s*#\s*/#/g' \
  | sed '$d' \
  > $fname.clean.txt

cat $fname.clean.txt \
  | head -n +400 \
  | awk -F'#' -v d2adir=${d2a_dir} -v a2ddir=${a2d_dir} -v prefix=${fname} \
    '{pair_id=$1;
      sub(/\.$/,"",$2);
      sub(/\.$/,"",$3);
      decoder=$2;
      answer=$3;
      printf "%s.\n%s.\n", decoder, answer > d2adir"/"prefix"_dec2ans_"pair_id".txt";
      printf "%s.\n%s.\n", answer, decoder > a2ddir"/"prefix"_ans2dec_"pair_id".txt";
     }'
