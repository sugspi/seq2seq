#!/bin/bash

conda create --yes -n py3 python=3
echo "source activate py3" > python_env.sh
chmod u+x python_env.sh
source python_env.sh
# Note: this installs different versions than those in requirements.txt
conda install -c anaconda tensorflow-gpu keras-gpu scikit-learn nomkl nltk=3.2.5 pydot graphviz
# Another possibility would be:
# conda create -n py3 --file requirements.txt

mkdir -p src/seq2seq/c2l
mkdir -p src/graph2seq/c2l

cd src/graph2seq
git clone https://github.com/pasmargo/graph-emb.git

