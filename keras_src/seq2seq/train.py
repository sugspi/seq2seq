# coding: utf-8
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

        return ([encoder_input_data, decoder_input_data, decoder_mask_matrix], decoder_target_data)

def attention():
    return 0

if __name__ == "__main__" :

    ###############################################################
    #   Setting Parameter
    ###############################################################
    batch_size = 64  # Batch size for training.
    epochs = 1  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    #num_samples = 10000  # Number of samples to train on.いらない？

    train_seq = EncDecSequence(corpus.all_input_formulas[8000:], corpus.all_target_texts[8000:], batch_size)
    val_seq = EncDecSequence(corpus.all_input_formulas[4000:8000], corpus.all_target_texts[4000:8000], batch_size)
    test_seq = EncDecSequence(corpus.all_input_formulas[:4000], corpus.all_target_texts[:4000], 1)

    ###############################################################
    #   evalのテスト．テスト終わり次第，一番下に持っていく
    ###############################################################
    #m_path = '/Users/guru/MyResearch/sg/keras_src/attention/models/'
    #model = load_model(m_path + 'elapsed_seq2seq.h5')
    #m = load_model(m_path + 'elapsed_seq2seq.h5') #kesu
    #m.save_weights(m_path + 'weights.h5')#kesu
    #model.load_weights(m_path + 'weights.h5')
    #encoder_model = load_model(m_path+'encoder.h5')
    #decoder_model = load_model(m_path+'decoder.h5')
    #eval.eval_blue(test_seq,model,encoder_model,decoder_model)
    ###############################################################

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

    decoder_dense = Dense(corpus.NUM_DECODER_TOKENS, activation='softmax',name='softmax2')
    new_decoder_outputs = decoder_dense(new_decoder_outputs)

    #mask_input = Input(shape=(corpus.MAX_DECODER_SEQ_LENGTH,corpus.NUM_DECODER_TOKENS), dtype='float32', name='mask_input')
    #new_decoder_outputs = multiply([mask_input, new_decoder_outputs])

    #callback function and parameter search
    #if you want to use below function, you add callbacks=[name-val]
    earlystop =keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
                #keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=0, mode='auto')
    #tensorboard = keras.callbacks.TensorBoard(log_dir='logs',write_images=True,write_graph=True,write_grads=True)
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
    raise
