#!/bin/bash

# USAGE: ./bicond.sh <prefix>
# prefix := tamf | tmf | taf | hamf | hmf | haf

prefix=$1

dec2ans_dir="en_results_${prefix}_dec2ans"
ans2dec_dir="en_results_${prefix}_ans2dec"

total=0
correct=0

for f in ${dec2ans_dir}/${prefix}_dec2ans*.candc.answer; do
  let total++
  base_filename=${f##*/}
  index=`echo $base_filename | awk -F'_' '{print $3}' | awk -F'.' '{print $1}'`
  d2a_answer=`head -1 $f`
  a2d_filename="${ans2dec_dir}/${base_filename/dec2ans/ans2dec}"
  a2d_answer=`head -1 ${a2d_filename}`
  if [ "${a2d_answer}" == "yes" ] && [ "${d2a_answer}" == "yes" ]; then
    let correct++
    bi_answer="yes"
  else
    bi_answer="unknown"
  fi
  echo -e $index"\t"$d2a_answer"\t"$a2d_answer"\t"$bi_answer
done > ans.tmp

sort -n ans.tmp > ans.index.tmp

cat ${prefix}.clean.txt | awk -F'#' '{print $2"\t"$3}' > sentences.tmp

paste ans.index.tmp sentences.tmp > table.tmp

echo -e "id\tdec=>ans\tans=>dec\tdec<=>ans\tdecoder sentence\tanswer sentence" \
  > ${prefix}.table

cat table.tmp >> ${prefix}.table

accuracy=`echo "scale=4; $correct / $total" | bc -l`
echo "Accuracy: "$correct" / "$total" = "$accuracy >> ${prefix}.score

rm ans.tmp ans.index.tmp sentences.tmp table.tmp
