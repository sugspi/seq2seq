import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="1", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

# coding: utf-8
#ifrom __future__ import print_function

from keras.models import Model
from keras.models import load_model
from keras.layers import Input, LSTM,Embedding, Dense, multiply, Flatten,Softmax,Reshape, Dropout
from keras.layers import Lambda,multiply,dot,concatenate,add
import keras
from keras import backend as K
# from keras.utils import plot_model

#from sklearn.model_selection import train_test_split

import numpy as np
import json
import re
import pydot

import nltk
from nltk.tree import Tree
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

from keras.utils import Sequence
import logging

from nltk.sem.logic import LogicParser
from nltk.sem.logic import LogicalExpressionException

batch_size = 64  # Batch size for training.
epochs = 45  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = '/gs/hs0/tgh-17IAH/manome/snli_0413_formula.txt'#'/home/8/17IA0973/snli_0122_graph.txt'
m_path = 'models13/'


#新１，機能語のリストを得る．
func_list = [] #機能語のリスト
f = open('func_word.txt')
line = f.readline()
while line:
    func_list.append(line.rstrip())
    line = f.readline()
f.close()

#新２，マスク行列の行を返す関数．
#入力の論理式の_から始まる述語をstemmginしたものとあらかじめstemmingしてある辞書を参考に1を立てる
def get_masking_list(inp, verbose=True):
    mask = np.zeros((1, max_decoder_seq_length,num_decoder_tokens),dtype='float32')
    for num,t in enumerate(inp) :

        if(t[:1]=='_'):
            ############### in_front_of の件、直しておきます…
            for s1 in t.split('_')[1:]:
                try:
                    for word in lem_dict[s1]:
                        index = target_token_index[word]
                        for i in range(max_decoder_seq_length):
                            mask[0,i,index] = 1.
                except:
                    if verbose: print("ERROR word : ",s1)
                    continue
    for f in func_index:
        for i in range(max_decoder_seq_length):
            mask[0,i,f] = 1.
    return mask


# Vectorize the data.
input_texts = []
target_texts = []
output_texts =[]
base_texts =[]
surf_texts =[]
input_characters = set()
target_characters = set()
lem_dict = dict()
lines = open(data_path)

for i, line in enumerate(lines):
    if i >= 50000: break
    line = line.split('#')
    input_text = line[0]
    target_text = line[1]
    ############# バグではないですが、target_textとかはすべて小文字にしておいたほうが良いです．
    ############# もし学習データに例えばParkとparkが存在すると語彙数が倍になってしまうので避けたく、前処理でよくやられてます．
    target_text = target_text.lstrip().lower()
    base_text = line[2].rstrip().lower()
    base_text = base_text.lstrip()
    #input_text = input_text.split(',')
    #input_text.append('EOS')

    input_text = re.sub('\(', ' ( ',input_text)
    input_text = re.sub('\)', ' ) ',input_text)
    input_text = re.split('\s|\.', input_text)
    input_text = [i for i in input_text if i not in ['TrueP', ''] ]
    input_text.append('EOS')
    input_texts.append(input_text)



    output_texts.append(target_text)
    surf_text = re.split('\s|\.', target_text)
    surf_texts.append(surf_text)

    target_text = 'BOS ' + target_text + ' EOS'
    target_text = [i for i in re.split('\s|\.', target_text) if i not in [''] ]
    target_texts.append(target_text)

    base_text = re.split('\s|\.', base_text)
    base_texts.append(base_text)


    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)


    for s,b in (zip(surf_text, base_text)):
        if not(b in lem_dict.keys()):
            lem_dict[b] = set()
        if not(s in lem_dict[b]):
            lem_dict[b].add(s)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters) + 1
num_decoder_tokens = len(target_characters) + 1
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i+1) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i+1) for i, char in enumerate(target_characters)])
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())# 0:tab,1:\n


class EncDecSequence(Sequence):
    def __init__(self, x, y, batch_size):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        encoder_input_data = np.zeros(
            (self.batch_size, max_encoder_seq_length),
            dtype='float32')

        decoder_input_data = np.zeros(
            (self.batch_size, max_decoder_seq_length),
            dtype='float32')

        decoder_target_data = np.zeros(
            (self.batch_size, max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')

        decoder_mask_matrix = np.zeros(
            (self.batch_size, max_decoder_seq_length,num_decoder_tokens),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(batch_x, batch_y)):
            decoder_mask_matrix[i] = get_masking_list(input_text, verbose=False)
            for t, char in enumerate(input_text):
                encoder_input_data[i, t] = input_token_index[char]
            for t, char in enumerate(target_text):
                decoder_input_data[i, t] = target_token_index[char]
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.

        return ([encoder_input_data, decoder_input_data], decoder_target_data)


#新３，機能語のインデックス格納
func_index = [target_token_index['EOS']]
for w in func_list:
    if w in target_token_index:
        func_index.append(target_token_index[w])


train_seq = EncDecSequence(input_texts[8000:], target_texts[8000:], batch_size)
val_seq = EncDecSequence(input_texts[4000:8000], target_texts[4000:8000], batch_size)
test_seq = EncDecSequence(input_texts[:4000], target_texts[:4000], 1)


# Define an input sequence and process it.
enc_main_input = Input(shape=(max_encoder_seq_length,), dtype='int32', name='enc_main_input')
encoder_inputs  = Embedding(output_dim=256, input_dim=num_encoder_tokens, mask_zero=True, input_length=max_encoder_seq_length,name='enc_embedding')(enc_main_input)
encoder_inputs = Dropout(0.5)(encoder_inputs)
encoder = LSTM(latent_dim, return_state=True,return_sequences=True,name='enc_lstm', dropout=0.5, recurrent_dropout=0.5)
encoder_outputs, state_h, state_c  = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
dec_main_input = Input(shape=(max_decoder_seq_length,), dtype='int32', name='dec_main_input')
decoder_inputs  = Embedding(output_dim=256, input_dim=num_decoder_tokens, mask_zero=True, input_length=max_decoder_seq_length,name='dec_embedding')(dec_main_input)
decoder_inputs  = Dropout(0.5)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,name='dec_lstm', dropout=0.5, recurrent_dropout=0.5)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
print("dec_hidden: ",K.int_shape(decoder_outputs))

inner_prod = dot([encoder_outputs,decoder_outputs], axes=2)
print("mul (enc,hid): ",K.int_shape(inner_prod))

a_vector = Softmax(axis=1)(inner_prod)
print("a_vecotr(softmax): ",K.int_shape(a_vector))

context_vector = dot([a_vector,encoder_outputs], axes=1)
print("context_vector: ",K.int_shape(context_vector))

concat_vector = concatenate([context_vector,decoder_outputs], axis=2)
print("concat_vector: ",K.int_shape(concat_vector))

decoder_tanh = Dense(latent_dim, activation='tanh',name='tanh')
new_decoder_outputs = decoder_tanh(concat_vector)
new_decoder_outputs  = Dropout(0.5)(new_decoder_outputs)
print("new_dec_hidden: ",K.int_shape(new_decoder_outputs))

decoder_dense = Dense(num_decoder_tokens, activation='softmax',name='softmax2')
new_decoder_outputs = decoder_dense(new_decoder_outputs)

### maskを新しくInputを追加
#mask_input = Input(shape=(max_decoder_seq_length,num_decoder_tokens), dtype='float32', name='mask_input')

#new_decoder_outputs = multiply([mask_input, new_decoder_outputs])

#callback function and parameter search
#if you want to use below function, you add callbacks=[name-val]
earlystop =keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
            #keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')
# tensorboard = keras.callbacks.TensorBoard(log_dir='logs',write_images=True,write_graph=True,write_grads=True)
checkpoint = keras.callbacks.ModelCheckpoint(
             filepath = m_path + 'elapsed_seq2seq.h5',
             monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

## Define the model that will turn
## `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([enc_main_input, dec_main_input], new_decoder_outputs)
## plot_model(model, to_file='model.png')
## Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


model.fit_generator(train_seq,
         epochs=epochs,
         validation_data=val_seq,
         use_multiprocessing=True,
         workers=3,
         callbacks=[checkpoint]
         )

# Save model
model = load_model(m_path + 'elapsed_seq2seq.h5')
m = load_model(m_path + 'elapsed_seq2seq.h5')
m.save_weights(m_path + 'weights.h5')
model.load_weights(m_path + 'weights.h5')

# encoder_model = load_model('encoder.h5')
# decoder_model = load_model('decoder.h5')

#encoder_outputs,_,_ = encoder(encoder_inputs)
encoder_model = Model(enc_main_input, [encoder_outputs]+encoder_states)
encoder_model.save(m_path + 'encoder.h5')

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
encoder_state_input_e = Input(shape=(max_encoder_seq_length, latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
_, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]

re_state_h = Reshape((1,256))(state_h)
inner_prod = dot([encoder_state_input_e,re_state_h], axes=2)
a_vector = Softmax(axis=1)(inner_prod)
context_vector = dot([a_vector,encoder_state_input_e], axes=1)
concat_vector = concatenate([context_vector,re_state_h], axis=2)
new_decoder_outputs = decoder_tanh(concat_vector)

decoder_outputs = decoder_dense(new_decoder_outputs)
#decoder_outputs = multiply([mask_input, decoder_outputs])

decoder_model = Model(
   [dec_main_input,encoder_state_input_e] + decoder_states_inputs,
   [decoder_outputs] + decoder_states)
decoder_model.save(m_path + 'decoder.h5')
#plot_model(decoder_model, to_file='decoder_model.png')

# Reverse-lookup token index to decode sequences back to
# something readable.

def decode_sequence(input_seq): ############# 新しく引数にinput_maskを追加
    # Encode the input as state vectors.
    e_state,h_state,c_state = encoder_model.predict(input_seq)
    states_value = [h_state,c_state]

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,max_decoder_seq_length))
    # print(input_seq)
    target_seq[0, 0] = target_token_index['BOS']
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq,e_state] + states_value)

        # Sample a token
        sampled_char = ''
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            decoded_sentence += '!'
        else:
            sampled_char = reverse_target_char_index[sampled_token_index]
            if sampled_char != 'EOS':
                decoded_sentence += sampled_char + ' '

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'EOS' or
           len(decoded_sentence) > max_decoder_seq_length + 15):
            stop_condition = True
            decoded_sentence = decoded_sentence.rstrip()
            decoded_sentence += '.'

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, max_decoder_seq_length))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


def remove_punct(text):
    if text.endswith('.'):
        return text[:-1]
    else:
        return text

#bleu evaluation
len_inp = len(test_seq)
sum_score = 0
results = []
for seq_index in range(len_inp):
    test_data, tmp = test_seq[seq_index]
    input_seq = test_data[0]
    #input_mask = test_data[2]
    decoded_sentence = decode_sequence(input_seq).lstrip()
    results.append(remove_punct(decoded_sentence).split(' '))
    fname = 'c2l13/result'+str(seq_index)+'.txt'
    f = open(fname, 'w')
    f.write(output_texts[seq_index]+'\n')
    f.write(decoded_sentence.strip()+'\n')
    f.close()
    #print('Input sentence:', input_texts[seq_index])
    #print('Decoded sentence:', decoded_sentence)
    #print('Answer sentence:', output_texts[seq_index])
    #print('')

bleu = corpus_bleu([[remove_punct(t).split(' ')] for t in output_texts], results)
print('bleu score',bleu)

