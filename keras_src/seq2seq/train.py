# coding: utf-8
import json
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="1", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

import keras
from keras.models import load_model, Model
from keras.layers import Input, LSTM, Embedding, Dense, Flatten, Softmax, Reshape, Dropout, Lambda, dot, concatenate, add, multiply
from keras.utils import Sequence

import txt_tool
import corpus
from decoder import decode_sequence, decode_sequence_with_mask
from masking import get_masking_vector
import eval

m_path = corpus.m_path

###############################################################
#   Setting Parameter
###############################################################
batch_size = 64  # Batch size for training.
epochs = 1  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.

###############################################################
#   callback function and parameter search
###############################################################
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
checkpoint = keras.callbacks.ModelCheckpoint(
             filepath = m_path + 'elapsed_seq2seq.h5',
             monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

#? batch_sizeについてあまり理解していない12/8なのでgenerationのときに，encoder_input_dataのbachを1にしてしまっている
class EncDecSequence(Sequence):
    def __init__(self, x, y, batch_size, model='attention' ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.model = model

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        encoder_input_data = np.zeros(
            (self.batch_size, corpus.MAX_ENCODER_SEQ_LENGTH),
            dtype='float32')

        decoder_input_data = np.zeros(
            (self.batch_size, corpus.MAX_DECODER_SEQ_LENGTH),
            dtype='float32')

        decoder_target_data = np.zeros(
            (self.batch_size, corpus.MAX_DECODER_SEQ_LENGTH, corpus.NUM_DECODER_TOKENS),
            dtype='float32')

        decoder_mask_matrix = np.zeros(
            (self.batch_size, corpus.MAX_DECODER_SEQ_LENGTH,corpus.NUM_DECODER_TOKENS),
            dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(batch_x, batch_y)):
            decoder_mask_matrix[i] = get_masking_vector(input_text, verbose=False)
            for t, token in enumerate(input_text):
                encoder_input_data[i, t] = corpus.formula_token_index[token]
            for t, token in enumerate(target_text):
                decoder_input_data[i, t] = corpus.target_token_index[token]
                if t > 0:
                    decoder_target_data[i, t - 1, corpus.target_token_index[token]] = 1.

        if(self.model == 'attention'):
            return ([encoder_input_data, decoder_input_data], decoder_target_data)
        elif(self.model == 'masking'):
            return ([encoder_input_data, decoder_input_data, decoder_mask_matrix], decoder_target_data)

        return 'cant create EncDecSequence'

def create_attention_model():
    # Define an input sequence and process it.
    enc_main_input = Input(shape=(corpus.MAX_ENCODER_SEQ_LENGTH,), dtype='int32', name='enc_main_input')
    encoder_inputs  = Embedding(output_dim=256, input_dim=corpus.NUM_ENCODER_TOKENS, mask_zero=True, input_length=corpus.MAX_ENCODER_SEQ_LENGTH,name='enc_embedding')(enc_main_input)
    encoder_inputs = Dropout(0.5)(encoder_inputs)
    encoder = LSTM(latent_dim, return_state=True,return_sequences=True,name='enc_lstm', dropout=0.5, recurrent_dropout=0.5)
    encoder_outputs, state_h, state_c  = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    dec_main_input = Input(shape=(corpus.MAX_DECODER_SEQ_LENGTH,), dtype='int32', name='dec_main_input')
    decoder_inputs  = Embedding(output_dim=256, input_dim=corpus.NUM_DECODER_TOKENS, mask_zero=True, input_length=corpus.MAX_DECODER_SEQ_LENGTH,name='dec_embedding')(dec_main_input)
    decoder_inputs  = Dropout(0.5)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,name='dec_lstm', dropout=0.5, recurrent_dropout=0.5)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
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

    decoder_dense = Dense(corpus.NUM_DECODER_TOKENS, activation='softmax',name='softmax2')
    new_decoder_outputs = decoder_dense(new_decoder_outputs)

    model = Model([enc_main_input, dec_main_input], new_decoder_outputs)

    ###############################################################
    #   Define encoder
    ###############################################################
    encoder_model = Model(enc_main_input, [encoder_outputs]+encoder_states)
    encoder_model.save(m_path + 'encoder.h5')

    ###############################################################
    #   Define decoder
    ###############################################################
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    encoder_state_input_e = Input(shape=(corpus.MAX_ENCODER_SEQ_LENGTH, latent_dim,))
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

    decoder_model = Model(
       [dec_main_input,encoder_state_input_e] + decoder_states_inputs,
       [decoder_outputs] + decoder_states)
    decoder_model.save(m_path + 'decoder.h5')

    return model

#def create_masking_model():


def train_model(train_seq, val_seq):
    #tensorboard = keras.callbacks.TensorBoard(log_dir='logs',write_images=True,write_graph=True,write_grads=True)
    # plot_model(model, to_file='model.png') #you need import pydot

    ###############################################################
    #   Setting Parameter
    ###############################################################
    if(corpus.model_name == 'attention'):
        model = create_attention_model()

    ## Run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit_generator(train_seq,
             epochs=epochs,
             validation_data=val_seq,
             use_multiprocessing=True,
             workers=3,
             callbacks=[checkpoint]
             )

    return model

if __name__ == "__main__":
    train_seq = EncDecSequence(corpus.all_input_formulas[200:], corpus.all_target_texts[200:], batch_size)
    val_seq = EncDecSequence(corpus.all_input_formulas[100:200], corpus.all_target_texts[100:200], batch_size)
    test_seq = EncDecSequence(corpus.all_input_formulas[:100], corpus.all_target_texts[:100], 1)

    train_model(train_seq, val_seq)
