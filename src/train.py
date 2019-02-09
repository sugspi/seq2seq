# coding: utf-8
import json
import pydot
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto(
#    gpu_options=tf.GPUOptions(
#        visible_device_list="1", # specify GPU number
#        allow_growth=True
#    )
#)
#set_session(tf.Session(config=config))

import keras
from keras.models import load_model, Model
from keras.layers import Input, LSTM, Embedding, Dense, Flatten, Softmax, Reshape, Dropout, Lambda, dot, concatenate, add, multiply
from keras.utils import Sequence, plot_model


import txt_tool
import corpus
from decoder import decode_sequence, decode_sequence_with_mask
from masking import get_masking_vector
import eval

m_path = corpus.m_path #later, I'll change getting arg in main func

###############################################################
#   Setting Parameter
###############################################################
param_json = open("param.json", 'r')#本来m_pathにb入ってる
params = json.load(param_json)
param_json.close()

batch_size = params['batch_size']  # Batch size for training.
epochs = params['epochs']  # Number of epochs to train for.
latent_dim = params['latent_dim']  # Latent dimensionality of the encoding space.
drop_out = params['drop_out']

###############################################################
#   callback function and parameter search
###############################################################
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
checkpoint = keras.callbacks.ModelCheckpoint(
             filepath = m_path + '{epoch:02d}-{val_loss:.2f}.h5',
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

        decoder_mask_matrix = np.full(
            (self.batch_size, corpus.MAX_DECODER_SEQ_LENGTH,corpus.NUM_DECODER_TOKENS),
            1e-6)

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

#if you want to see the shape of layer, use K.int_shape() function.
def create_attention_model():
    # Define an input sequence and process it embeddingのここもテストするべきではないか.
    enc_main_input = Input(shape=(corpus.MAX_ENCODER_SEQ_LENGTH,), dtype='int32', name='enc_main_input')
    encoder_inputs  = Embedding(output_dim=256, input_dim=corpus.NUM_ENCODER_TOKENS, mask_zero=True, input_length=corpus.MAX_ENCODER_SEQ_LENGTH, name='enc_embedding')(enc_main_input)
    encoder_inputs = Dropout(drop_out)(encoder_inputs)
    encoder = LSTM(latent_dim, return_state=True,return_sequences=True, name='enc_lstm', dropout=drop_out, recurrent_dropout=drop_out)
    encoder_outputs, state_h, state_c  = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    dec_main_input = Input(shape=(corpus.MAX_DECODER_SEQ_LENGTH,), dtype='int32', name='dec_main_input')
    decoder_inputs  = Embedding(output_dim=256, input_dim=corpus.NUM_DECODER_TOKENS, mask_zero=True, input_length=corpus.MAX_DECODER_SEQ_LENGTH, name='dec_embedding')(dec_main_input)
    decoder_inputs  = Dropout(drop_out)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,name='dec_lstm', dropout=drop_out, recurrent_dropout=drop_out)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    inner_prod = dot([encoder_outputs, decoder_outputs], axes=2)
    a_vector = Softmax(axis=1)(inner_prod)
    context_vector = dot([a_vector, encoder_outputs], axes=1)
    concat_vector = concatenate([context_vector, decoder_outputs], axis=2)
    decoder_tanh = Dense(latent_dim, activation='tanh', name='tanh')
    new_decoder_outputs = decoder_tanh(concat_vector)
    new_decoder_outputs  = Dropout(drop_out)(new_decoder_outputs)

    decoder_dense = Dense(corpus.NUM_DECODER_TOKENS, activation='softmax', name='softmax2')
    new_decoder_outputs = decoder_dense(new_decoder_outputs)

    model = Model([enc_main_input, dec_main_input], new_decoder_outputs)

    ###############################################################
    #   Define encoder
    ###############################################################
    encoder_model = Model(enc_main_input, [encoder_outputs] + encoder_states)
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
    inner_prod = dot([encoder_state_input_e, re_state_h], axes=2)
    a_vector = Softmax(axis=1)(inner_prod)
    context_vector = dot([a_vector,encoder_state_input_e], axes=1)
    concat_vector = concatenate([context_vector,re_state_h], axis=2)
    new_decoder_outputs = decoder_tanh(concat_vector)
    decoder_outputs = decoder_dense(new_decoder_outputs)

    decoder_model = Model(
       [dec_main_input,encoder_state_input_e] + decoder_states_inputs,
       [decoder_outputs] + decoder_states)
    decoder_model.save(m_path + 'decoder.h5')

    plot_model(model, to_file= m_path + 'model.png')
    plot_model(encoder_model, to_file= m_path + 'encoder_model.png')
    plot_model(decoder_model, to_file= m_path + 'decoder_model.png')

    return model

def create_masking_model():
    # Define an input sequence and process it.
    enc_main_input = Input(shape=(corpus.MAX_ENCODER_SEQ_LENGTH,), dtype='int32', name='enc_main_input')
    encoder_inputs  = Embedding(output_dim=256, input_dim=corpus.NUM_ENCODER_TOKENS, mask_zero=True, input_length=corpus.MAX_ENCODER_SEQ_LENGTH,name='enc_embedding')(enc_main_input)
    encoder_inputs = Dropout(drop_out)(encoder_inputs)
    encoder = LSTM(latent_dim, return_state=True,return_sequences=True,name='enc_lstm', dropout=drop_out, recurrent_dropout=drop_out)
    encoder_outputs, state_h, state_c  = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    dec_main_input = Input(shape=(corpus.MAX_DECODER_SEQ_LENGTH,), dtype='int32', name='dec_main_input')
    decoder_inputs  = Embedding(output_dim=256, input_dim=corpus.NUM_DECODER_TOKENS, mask_zero=True, input_length=corpus.MAX_DECODER_SEQ_LENGTH,name='dec_embedding')(dec_main_input)
    decoder_inputs  = Dropout(drop_out)(decoder_inputs)
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,name='dec_lstm', dropout=drop_out, recurrent_dropout=drop_out)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)

    inner_prod = dot([encoder_outputs,decoder_outputs], axes=2)
    a_vector = Softmax(axis=1)(inner_prod)
    context_vector = dot([a_vector,encoder_outputs], axes=1)
    concat_vector = concatenate([context_vector,decoder_outputs], axis=2)
    decoder_tanh = Dense(latent_dim, activation='tanh',name='tanh')
    new_decoder_outputs = decoder_tanh(concat_vector)
    new_decoder_outputs  = Dropout(drop_out)(new_decoder_outputs)

    decoder_dense = Dense(corpus.NUM_DECODER_TOKENS, activation='softmax',name='softmax2')
    new_decoder_outputs = decoder_dense(new_decoder_outputs)

    ### to add masking vector###
    mask_input = Input(shape=(corpus.MAX_DECODER_SEQ_LENGTH,corpus.NUM_DECODER_TOKENS), dtype='float32', name='mask_input')
    new_decoder_outputs = multiply([mask_input, new_decoder_outputs])
    model = Model([enc_main_input, dec_main_input, mask_input], new_decoder_outputs)

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

    re_state_h = Reshape((1,latent_dim))(state_h)
    inner_prod = dot([encoder_state_input_e,re_state_h], axes=2)
    a_vector = Softmax(axis=1)(inner_prod)
    context_vector = dot([a_vector,encoder_state_input_e], axes=1)
    concat_vector = concatenate([context_vector,re_state_h], axis=2)
    new_decoder_outputs = decoder_tanh(concat_vector)

    decoder_outputs = decoder_dense(new_decoder_outputs)
    decoder_outputs = multiply([mask_input, decoder_outputs])

    decoder_model = Model(
       [dec_main_input, encoder_state_input_e, mask_input] + decoder_states_inputs,
       [decoder_outputs] + decoder_states)
    decoder_model.save(m_path + 'decoder.h5')

    plot_model(model, to_file= m_path + 'model.png')
    plot_model(encoder_model, to_file= m_path + 'encoder_model.png')
    plot_model(decoder_model, to_file= m_path + 'decoder_model.png')

    return model


def train_model(train_seq, val_seq):
    #tensorboard = keras.callbacks.TensorBoard(log_dir='logs',write_images=True,write_graph=True,write_grads=True)
    # plot_model(model, to_file='model.png') #you need import pydot

    ###############################################################
    #   Setting Parameter
    ###############################################################
    if(corpus.model_name == 'attention'):
        model = create_attention_model()
    elif(corpus.model_name == 'masking'):
        model = create_masking_model()

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

    if(corpus.model_name == 'attention'):
        train_seq = EncDecSequence(corpus.all_input_formulas[275:], corpus.all_target_texts[275:], batch_size)
        val_seq = EncDecSequence(corpus.all_input_formulas[137:275], corpus.all_target_texts[137:275], batch_size)
        test_seq = EncDecSequence(corpus.all_input_formulas[:137], corpus.all_target_texts[:137], 1)

    elif(corpus.model_name == 'masking'):
        train_seq = EncDecSequence(corpus.all_input_formulas[5077:], corpus.all_target_texts[5077:], batch_size, 'masking')
        val_seq = EncDecSequence(corpus.all_input_formulas[2640:5077], corpus.all_target_texts[2640:7077], batch_size, 'masking')
        test_seq = EncDecSequence(corpus.all_input_formulas[:2640], corpus.all_target_texts[:2640], 1, 'masking')


    ###############################################################
    #   print files
    ###############################################################
    fname = corpus.m_path + 'setting.txt'
    f = open(fname, 'w')
    f.write("NUM_ENCODER_TOKENS: " + str(corpus.NUM_ENCODER_TOKENS) + '\n')
    f.write("NUM_DECODER_TOKENS: " + str(corpus.NUM_DECODER_TOKENS) + '\n')
    f.write("MAX_ENCODER_SEQ_LENGTH: " + str(corpus.MAX_ENCODER_SEQ_LENGTH) + '\n')
    f.write("MAX_DECODER_SEQ_LENGTH: " + str(corpus.MAX_DECODER_SEQ_LENGTH) + '\n')
    f.close()
    train_model(train_seq, val_seq)
