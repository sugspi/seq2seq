import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

# coding: utf-8
#ifrom __future__ import print_function

from keras.models import Model
from keras.models import load_model
from keras.layers import Input, LSTM,Embedding, Dense, Permute, Flatten,Softmax
from keras.layers import Lambda,multiply,dot,concatenate,add
import keras

from keras import backend as K

#from sklearn.model_selection import train_test_split

import numpy as np
import json
import re
import math
import pydot
from keras.utils import plot_model

import nltk
from nltk.tree import Tree
from nltk.translate.bleu_score import sentence_bleu

####

import logging

from nltk.sem.logic import LogicParser
from nltk.sem.logic import LogicalExpressionException


batch_size = 256  # Batch size for training.
epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path =  '/Users/guru/MyResearch/sg/data/snli/text_graph/small.txt'

# Vectorize the data.
input_texts = []
target_texts = []
output_texts =[]
input_characters = set()
target_characters = set()
lines = open(data_path)


for line in lines:
    line = line.split('#')
    input_text = line[0]
    target_text = line[1]
    input_text = input_text.split(',')
    input_text.append('EOS')
    output_texts.append(target_text.lstrip())
    target_text = 'BOS' + target_text + 'EOS'
    target_text = re.split('\s|\.', target_text)
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
print(len(output_texts))
print(len(target_texts))

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


encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length),
    dtype='float32')

decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length),
    dtype='float32')

decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t] = input_token_index[char]
    for t, char in enumerate(target_text):
        decoder_input_data[i, t] = target_token_index[char]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

test_input_data =  encoder_input_data[:1500]
output_texts = output_texts[:1500]
#encoder_input_data = np.delete(encoder_input_data,[i for i in range(1500)],0)
#decoder_input_data = np.delete(decoder_input_data,[i for i in range(1500)],0)
#decoder_target_data = np.delete(decoder_target_data,[i for i in range(1500)],0)

print("test: ",len(encoder_input_data))
print("inp: ",len(decoder_input_data))
print("out: ",len(decoder_target_data))


#mask_zero=Trueをemnbeddingレイヤーで消してしまった
# Define an input sequence and process it.
enc_main_input = Input(shape=(max_encoder_seq_length,), dtype='int32', name='enc_main_input')
encoder_inputs  = Embedding(output_dim=256, input_dim=num_encoder_tokens, input_length=max_encoder_seq_length,name='enc_embedding')(enc_main_input)
encoder = LSTM(latent_dim, return_state=True,return_sequences=True,name='enc_lstm')
encoder_outputs, state_h, state_c  = encoder(encoder_inputs)
encoder_states = [state_h, state_c]
print("enc_hidden: ",K.int_shape(encoder_outputs))

# Set up the decoder, using `encoder_states` as initial state.
dec_main_input = Input(shape=(max_decoder_seq_length,), dtype='int32', name='dec_main_input')
decoder_inputs  = Embedding(output_dim=256, input_dim=num_decoder_tokens, input_length=max_decoder_seq_length,name='dec_embedding')(dec_main_input)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,name='dec_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
print("dec_hidden: ",K.int_shape(decoder_outputs))


inner_prod = dot([encoder_outputs,decoder_outputs], axes=2)
print("mul (enc,hid): ",K.int_shape(inner_prod))

################ Denseだと内積の結果にさらに行列をかけてしまうのでちょっと違うかも
################ keras.layers.Softmaxというのがあるらしいですよ
a_vector = Dense(max_decoder_seq_length, activation='softmax',name='softmax')
a_vector = a_vector(inner_prod)
print("a_vecotr(softmax): ",K.int_shape(a_vector))

context_vector = dot([a_vector,encoder_outputs], axes=1)
print("context_vector: ",K.int_shape(context_vector))

# concat_vector = add([context_vector,decoder_outputs])#, axis=-1)
concat_vector = concatenate([context_vector,decoder_outputs], axis=2) ################# addじゃなくてconcatenate
print("concat_vector: ",K.int_shape(concat_vector))

decoder_tanh = Dense(latent_dim, activation='tanh',name='tanh')
new_decoder_outputs = decoder_tanh(concat_vector)
print("new_dec_hidden: ",K.int_shape(new_decoder_outputs))

############### エラーの原因は、上の(バッチ,出力系列,256)のshapeのdecoder_tanh(本でのhチルダ)が一番最後になってたことっぽいです．
############### 一番最後の出力は(バッチ,出力系列,num_decoder_tokens)のテンソルがほしくて、
############### というのも一番最後の出力のテンソルが語彙数次元(num_decoder_tokens)ないと、
############### categorical_crossentropyとかが計算できなくなってしまいます．
############### 本でhチルダを計算した後にどうしてたか知らないのですが、もう一度Denseして
############### (バッチ,出力系列,num_decoder_tokens)にすると動きました．
#new_decoder_outputs = Dense(num_decoder_tokens)(new_decoder_outputs)

decoder_dense = Dense(num_decoder_tokens, activation='softmax',name='softmax2')
new_decoder_outputs = decoder_dense(new_decoder_outputs)

#callback function and parameter search
#if you want to use below function, you add callbacks=[name-val]
earlystop =keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
            #keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')
tensorboard = keras.callbacks.TensorBoard(log_dir='logs',write_images=True,write_graph=True,write_grads=True)
checkpoint = keras.callbacks.ModelCheckpoint(
             filepath = 'elapsed_seq2seq.h5',#'seq2seq_model{epoch:02d}-loss{loss:.2f}-vloss{val_loss:.2f}.h5',
             monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([enc_main_input, dec_main_input], new_decoder_outputs)
model.summary()

plot_model(model, to_file='model.png')
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
#model = load_model('elapsed_seq2seq.h5')
#m = load_model('elapsed_seq2seq.h5')
#m.save_weights('weights.h5')
#model.load_weights('weights.h5')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[checkpoint,tensorboard]
          )

# Save model
model.save('s2s.h5')
model = load_model('elapsed_seq2seq.h5')
m = load_model('elapsed_seq2seq.h5')
m.save_weights('weights.h5')
model.load_weights('weights.h5')

#encoder_model = load_model('encoder.h5')
#decoder_model = load_model('decoder.h5')

encoder_model = Model(enc_main_input, encoder_states)
encoder_model.save('encoder.h5')

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [dec_main_input] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
decoder_model.save('decoder.h5')

# Reverse-lookup token index to decode sequences back to
# something readable.

reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())# 0:tab,1:\n

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,max_decoder_seq_length))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['BOS']
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
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

#bleu evaluation
len_inp = len(test_input_data)
sum_score = 0
for seq_index in range(len_inp-1):
    input_seq = test_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq).lstrip()
    sum_score += sentence_bleu([output_texts[seq_index]],decoded_sentence)
    fname = 'c2l/result'+str(seq_index)+'.txt'
    f = open(fname, 'w')
    f.write(output_texts[seq_index])
    f.write(decoded_sentence.strip()+'\n')
    f.close()
    #print('Input sentence:', input_texts[seq_index])
    #if (seq_index%100) == 0 :
    #    print('Decoded sentence:', decoded_sentence)
    #    print('Answer sentence:', output_texts[seq_index])
print('bleu score',(sum_score/len_inp))
