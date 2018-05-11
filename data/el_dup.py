import numpy as np
import json
import re
import glob


import nltk
from nltk.tree import Tree
from nltk.translate.bleu_score import sentence_bleu


import logging


for fname in glob.glob('/Users/guru/MyResearch/sg/data/tmp/t/*.txt'):
    print (fname)
    fname = open(fname, 'r')
    f = open('snli_0428.txt', 'a')

    data = []

    jdict = json.load(fname)
    keys = [int(i) for i in list(jdict.keys())]
    keys.sort()
    keys = [str(i) for i in keys]
    for line in keys:
        input_text = (jdict[line])['formula']
        target_text = (jdict[line])['text'].rstrip()
        base_text = (jdict[line])['base'].rstrip()
        data.append(input_text + '#' + target_text + '#' + base_text)

    for i in data :
        f.write(str(i)+'\n')
    f.close()
    print(len(data))
