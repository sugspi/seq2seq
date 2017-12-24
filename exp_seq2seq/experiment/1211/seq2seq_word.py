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
epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = '/Users/guru/MyResearch/sg/snli/json/snli_input_data_100_1214.json'

# Vectorize the data.
input_formulas = []
target_texts = []
input_symbols = set()
target_words = set()
lines = open(data_path)

#json ver
jdict = json.load(lines)

for line in jdict:
    input_formula = (jdict[line])['formula']
    input_formula = re.sub('\(', '( ',input_formula)
    input_formula = re.sub('\)', ' )',input_formula)
    input_formula = re.split('\s|\.', input_formula)
    target_text = (jdict[line])['text']
    target_text = 'BOS ' + target_text + ' EOS'
    target_text = re.split('\s|\.', target_text)
    input_formulas.append(input_formula)
    target_texts.append(target_text)

    for symbol in input_formula:
        if symbol not in input_symbols:
            input_symbols.add(symbol)
    for word in target_text:
        if word not in target_words:
            target_words.add(word)

input_symbols = sorted(list(input_symbols))
target_words = sorted(list(target_words))
num_encoder_tokens = len(input_symbols)
num_decoder_tokens = len(target_words)
max_encoder_seq_length = max([len(txt) for txt in input_formulas])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_formulas))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(symbol, i) for i, symbol in enumerate(input_symbols)])
target_token_index = dict(
    [(word, i) for i, word in enumerate(target_words)])


encoder_input_data = np.zeros(
    (len(input_formulas), max_encoder_seq_length),
    dtype='float32')

decoder_input_data = np.zeros(
    (len(target_texts), max_decoder_seq_length),
    dtype='float32')

#(1)これだけindex行列ではなくone-hotな理由がわかっていない．indexの方がよいのか？
decoder_target_data = np.zeros(
    (len(input_formulas), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_formula, target_text) in enumerate(zip(input_formulas, target_texts)):
    for t, symbol in enumerate(input_formula):
        encoder_input_data[i, t] = input_token_index[symbol]
    for t, word in enumerate(target_text):
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.

enc_main_input = Input(shape=(max_encoder_seq_length,), dtype='int32', name='enc_main_input')
encoder_inputs  = Embedding(output_dim=512, input_dim=num_encoder_tokens, input_length=max_encoder_seq_length)(enc_main_input)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c  = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

dec_main_input = Input(shape=(max_decoder_seq_length,), dtype='int32', name='dec_main_input')
decoder_inputs  = Embedding(output_dim=512, input_dim=num_decoder_tokens, input_length=max_decoder_seq_length)(dec_main_input)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

#callback function and parameter search
earlystop =keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
tensorboard = keras.callbacks.TensorBoard(log_dir='logs',write_images=True,write_graph=True,)
checkpoint = keras.callbacks.ModelCheckpoint(
             filepath = 'elapsed_seq2seq.h5',#'seq2seq_model{epoch:02d}-loss{loss:.2f}-vloss{val_loss:.2f}.h5',
             monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model = Model([enc_main_input, dec_main_input], decoder_outputs)
#plot_model(model, to_file='model.png')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
          #callbacks=[tensorboard,checkpoint])

model.save('s2s.h5')


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

reverse_target_word_index = dict(
    (i, word) for word, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    #(2)なぜ長さ１のsequenceが必要？そしてこのshapeでよいか？
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
        #(3)なぜoutput_tokens[0, -1, :]の行をとるのか？
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_word_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 'EOS' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, max_decoder_seq_length))
        #(4) (2)と同じ疑問だがtarget_seqはこのshpaeでよいのか？？
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

for seq_index in range(100):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_formulas[seq_index])
    print('Decoded sentence:', decoded_sentence)
