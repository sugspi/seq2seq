import numpy as np
import json
import re
#import pydot

import nltk
from nltk.tree import Tree
from nltk.translate.bleu_score import sentence_bleu

####

import logging



# Path to the data txt file on disk.
data_path =  '1.txt'

# Vectorize the data.
input_texts = []
target_texts = []
base_texts =[]
input_characters = set()
target_characters = set()
lines = open(data_path)

set_fom = set()
set_fom2 = set()
set_txt = set()
c1 = 0
c2 = 0
c3 = 0

f = open('snli_0408_10000.txt', 'a')

data = []

jdict = json.load(lines)
keys = [int(i) for i in list(jdict.keys())]
keys.sort()
keys = [str(i) for i in keys]
for line in keys:
    input_text = (jdict[line])['formula']
    target_text = (jdict[line])['text'].rstrip()
    base_text = (jdict[line])['base'].rstrip()
    if  target_text not in set_txt:
        set_txt.add(target_text)
        data.append(input_text + '#' + target_text + '#' + base_text)

for i in data :
    f.write(str(i)+'\n')
f.close()



print(len(data))
