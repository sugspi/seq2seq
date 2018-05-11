#!/bin/bash

# Usage: 
#
# ./en/eval_gen.sh <prefix> <ncores> <coqlib:DEFAULT=coqlib_sick.v>
#
# Example:
#
# ./en/eval_gen.sh amf_dec2ans 10 
#
# <prefix> ::= <amf_dec2ans> <amf_ans2dec>

prefix=$1
cores=$2
coqlib=${3:-'./en/coqlib_sick.v'}
nbest="1"
templates="semantic_templates_en_event.yaml"

plain_dir="en_plain"
parsed_dir="en_parsed"
results_dir="en_results"
mkdir -p $plain_dir $results_dir

ls -v ${plain_dir}/${prefix}_* > ${plain_dir}/${prefix}.files

ndata=`cat ${plain_dir}/${prefix}.files | wc -l`
lines_per_split=`python -c "from math import ceil; print(int(ceil(float(${ndata})/${cores})))"`

rm -f ${plain_dir}/${prefix}.files_??
split -l $lines_per_split ${plain_dir}/${prefix}.files ${plain_dir}/${prefix}.files_

# Copy a coq static library and compile it
cp en/coqlib_sick.v coqlib.v
coqc coqlib.v
cp en/tactics_coq_fracas.txt tactics_coq.txt

# Run pipeline for each entailment problem.
for ff in ${plain_dir}/${prefix}.files_??; do
  for f in `cat ${ff}`; do
    ./en/rte_en_mp_any.sh $f en/$templates $nbest
  done &
done

# Wait for the parallel processes to finish.
wait

total=0
correct=0
for f in ${plain_dir}/${prefix}_*.tok; do
  let total++
  base_filename=${f##*/}
  sys_filename=${results_dir}/${base_filename/.tok/.answer}
  gold_answer="yes"
  if [ ! -e ${sys_filename} ]; then
    sys_answer="unknown"
  else
    sys_answer=`head -1 ${sys_filename}`
    if [ ! "${sys_answer}" == "unknown" ] && [ ! "${sys_answer}" == "yes" ] && [ ! "${sys_answer}" == "no" ]; then
      sys_answer="unknown"
    fi
  fi
  if [ "${gold_answer}" == "${sys_answer}" ]; then
    let correct++
  fi
  echo -e $f"\t"$gold_answer"\t"$sys_answer
done > $prefix.table

accuracy=`echo "scale=4; $correct / $total" | bc -l`
echo "Accuracy: "$correct" / "$total" = "$accuracy > $prefix.score
