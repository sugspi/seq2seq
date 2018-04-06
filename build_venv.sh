#!/bin/bash

# Checks that the virtualenv exists. If it doesn't, it automatically creates it.
if [ ! -d "./py3" ]; then
    echo "Creating a python2 virtual environment locally";
    virtualenv -p /usr/bin/python3 py3;
fi

echo "Activating python3 virtual environment"
source py3/bin/activate;

# Runs pip to install the dependencies from requirements.pip into the virtualenv. 
# For maximum efficiency, each time the installation succeeds, the file ve/updated is touched. 
# On subsequent runs, the script knows to skip installation unless the modification date of requirements.txt is more recent than that of requirements.txt.
if [ ! -f "./py3/updated" -o ./requirements.txt -nt ./py3/updated ]; then
    pip install -r ./requirements.txt
    touch ./py3/updated
    echo "Required dependencies installed."
fi

mkdir -p src/seq2seq/c2l
mkdir -p src/graph2seq/c2l

cd src/graph2seq
git clone https://github.com/pasmargo/graph-emb.git
