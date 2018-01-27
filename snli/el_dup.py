import numpy as np
import json
import re
#import pydot

import nltk
from nltk.tree import Tree
from nltk.translate.bleu_score import sentence_bleu

####

import logging


batch_size = 256  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path =  '/Users/guru/MyResearch/sg/snli/json/snli_input_data_1214.json'

# Vectorize the data.
input_texts = []
target_texts = []
output_texts =[]
input_characters = set()
target_characters = set()
lines = open(data_path)

set_fom = set()
set_fom2 = set()
set_txt = set()
c1 = 0
c2 = 0
c3 = 0

f = open('snli_0118.txt', 'w')

data = []

jdict = json.load(lines)
keys = [int(i) for i in list(jdict.keys())]
keys.sort()
keys = [str(i) for i in keys]
for line in keys:
    input_text = (jdict[line])['formula']
    target_text = (jdict[line])['text'].rstrip()
    if  target_text not in set_txt:
        set_txt.add(target_text)
        data.append(input_text + '#' + target_text)

for i in data :
    f.write(str(i)+'\n')
f.close()



print(len(data))
print(len(set_txt))
raise
