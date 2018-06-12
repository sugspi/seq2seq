#!/bin/bash

# USAGE: ./exp_entailment.sh <prefix> <number of cores> <option>
#
# <prefix> = graph | token | hts (hypothesis, treebank semantics)
# <option> = mask

# ./exp_entailment.sh hts 35
# ./exp_entailment.sh graph 35
# ./exp_entailment.sh graph 35 mask
# ./exp_entailment.sh token 35
# ./exp_entailment.sh token 35 mask

prefix=$1
ncore=$2
option=${3:-""}

# # Extract problems
# if [ ! -d "en_plain_${prefix}" ]; then
#   ./extract_snli.sh ${prefix}.txt
# fi

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

# prediction => answer
mkdir en_plain
cp en_plain_${prefix}_${option}/${prefix}_pred2ans/* en_plain/

./en/eval_gen.sh ${prefix}_pred2ans $ncore

wait

mv en_parsed en_parsed_${prefix}_pred2ans
mv en_results en_results_${prefix}_pred2ans
rm -rf en_plain

# answer => prediction
mkdir en_plain
cp en_plain_${prefix}_${option}/${prefix}_ans2pred/* en_plain/

./en/eval_gen.sh ${prefix}_ans2pred $ncore

wait

mv en_parsed en_parsed_${prefix}_ans2pred
mv en_results en_results_${prefix}_ans2pred
rm -rf en_plain

./bicond_eval.sh $prefix $option

mkdir gen_${prefix}_${option}
mv ${prefix}* en_plain_${prefix}_${option} en_parsed_${prefix}* en_results_${prefix}* gen_${prefix}_${option}
