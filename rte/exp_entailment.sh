#!/bin/bash

# USAGE: ./exp_entailment.sh <prefix> <number of cores>
#
# <prefix> = tamf | tmf | taf | hamf | hmf | haf
#

prefix=$1
ncore=$2

# Extract problems
if [ ! -d "en_plain_${prefix}" ]; then
  ./extract_snli.sh ${prefix}.txt
fi

if [ -d "en_plain" ]; then
  echo "en_plain directory already exists"
  exit 1
fi

if [ -d "en_parsed" ]; then
  echo "en_parsed directory already exists"
  exit 1
fi

if [ -d "en_results" ]; then
  echo "en_results directory already exists"
  exit 1
fi

# decoder => answer
mkdir en_plain
cp en_plain_${prefix}/${prefix}_dec2ans/* en_plain/

./en/eval_gen.sh ${prefix}_dec2ans $ncore

wait

mv en_parsed en_parsed_${prefix}_dec2ans
mv en_results en_results_${prefix}_dec2ans
rm -rf en_plain

# answer => decoder
mkdir en_plain
cp en_plain_${prefix}/${prefix}_ans2dec/* en_plain/

./en/eval_gen.sh ${prefix}_ans2dec $ncore

wait

mv en_parsed en_parsed_${prefix}_ans2dec
mv en_results en_results_${prefix}_ans2dec
rm -rf en_plain

./bicond_eval.sh $prefix

mkdir gen_$prefix
mv ${prefix}* en_plain_${prefix} en_parsed_${prefix}* en_results_${prefix}* gen_$prefix
