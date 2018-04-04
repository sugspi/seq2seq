#!/bin/bash

conda create --yes -n py3 python=3 flask
echo "source activate py3" > python_env.sh
chmod u+x python_env.sh
source python_env.sh
# Note: this installs different versions than those in requirements.txt
conda install -c anaconda tensorflow-gpu keras-gpu scikit-learn nomkl
# Another possibility would be:
# conda create -n py3 --file requirements.txt
