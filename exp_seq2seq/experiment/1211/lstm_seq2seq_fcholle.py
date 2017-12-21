from __future__ import print_function

from keras.models import Model
from keras.models import load_model
from keras.layers import Input, LSTM,Embedding,Dense
import keras
from keras.utils.vis_utils import plot_model

import numpy as np
import json
import h5py
import re

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = '/Users/guru/MyResearch/sg/snli/json/snli_input_data_100_1214.json'


#load dictionary
#dictionary = corpora.Dictionary.load_from_text('livedoordic.txt')

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
lines = open(data_path)

#json ver
jdict = json.load(lines)

for line in jdict:
    input_text = (jdict[line])['formula']
    input_text = re.sub('\(', '( ',input_text)
    input_text = re.sub('\)', ' )',input_text)
    input_text = re.split('\s|\.', input_text)
    target_text = (jdict[line])['text']
    target_text = 'BOS ' + target_text + ' EOS'
    target_text = re.split('\s|\.', target_text)
    input_texts.append(input_text)
    target_texts.append(target_text)
    #print(input_text)
    #print(target_text,"\n")

    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)


#print("input len",len(input_texts))
#print("output len",len(target_texts))

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

#np.zeros 0を要素とする配列(shape, dtype = float, order = ‘C’)　３次元
#encoder_input_data = np.zeros(
#    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
#    dtype='float32')
#decoder_input_data = np.zeros(
#    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
#    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
#    for t, char in enumerate(input_text):
#        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
#        decoder_input_data[i, t, target_token_index[char]] = 1.
       if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length),
    dtype='float32')

decoder_input_data = np.zeros(
    (len(target_texts), max_decoder_seq_length),
    dtype='float32')

#decoder_target_data = np.zeros(
#    (len(target_texts), max_decoder_seq_length),
#    dtype='float32')


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t] = input_token_index[char]
    for t, char in enumerate(target_text):
        decoder_input_data[i, t] = target_token_index[char]
        #if t > 0:
        #    decoder_target_data[i, t - 1] = target_token_index[char]

# Define an input sequence and process it.

enc_main_input = Input(shape=(max_encoder_seq_length,), dtype='int32', name='enc_main_input')
encoder_inputs  = Embedding(output_dim=512, input_dim=num_encoder_tokens, input_length=max_encoder_seq_length)(enc_main_input)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c  = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
dec_main_input = Input(shape=(max_decoder_seq_length,), dtype='int32', name='dec_main_input')
decoder_inputs  = Embedding(output_dim=512, input_dim=num_decoder_tokens, input_length=max_decoder_seq_length)(dec_main_input)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

#callback function and parameter search
earlystop =keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
            #keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')
tensorboard = keras.callbacks.TensorBoard(log_dir='logs',write_images=True,write_graph=True,)

checkpoint = keras.callbacks.ModelCheckpoint(
             filepath = 'elapsed_seq2seq.h5',#'seq2seq_model{epoch:02d}-loss{loss:.2f}-vloss{val_loss:.2f}.h5',
             monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

#トレーニングデータ，テストデータ分割
#data_train, data_test, label_train, label_test = train_test_split(bow, labels_list, test_size=0.5)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`

model = Model([enc_main_input, dec_main_input], decoder_outputs)
#plot_model(model, to_file='model.png')
# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
          #callbacks=[tensorboard,checkpoint])

# Save model
model.save('s2s.h5')

# choose input data version
 #input_texts_inf = []
#lines = open('predict.txt').read().split('\n')

#for line in lines[: min(num_samples, len(lines) - 1)]:
#    input_text_inf = line
#    input_texts_inf.append(input_text_inf)

#print('Number of samples:', len(input_texts_inf))


#np.zeros 0を要素とする配列(shape, dtype = float, order = ‘C’)　３次元
#encoder_input_data_inf = np.zeros(
#    (len(input_texts_inf), max_encoder_seq_length, num_encoder_tokens),
#    dtype='float32')


#for i, input_text_inf in enumerate(input_texts_inf):
#    for t, char in enumerate(input_text_inf):
#        encoder_input_data_inf[i, t, input_token_index[char]] = 1.


#model = load_model('s2s.h5')

#encoder_model = load_model('encoder.h5')
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.save('encoder.h5')


#decoder_model = load_model('decoder.h5')
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
decoder_model.save('decoder.h5')

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())

reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['BOS']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence



for seq_index in range(10):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    #input_seq = encoder_input_data[seq_index: seq_index + 1]
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
