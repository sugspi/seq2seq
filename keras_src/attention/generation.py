# coding: utf-8
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model

import txt_tool
import corpus
from decoder import decode_sequence

m_path = 'models/'

def surface_realization(formula, model, encoder_model, decoder_model):
    lst_formula = txt_tool.formula_to_list(formula)
    encoder_input_data = np.zeros((1, corpus.MAX_ENCODER_SEQ_LENGTH),dtype='float32')

    for t, token in enumerate(formula):
        if token in corpus.formula_token_index:
            encoder_input_data[0,t] = corpus.formula_token_index[token]

    decoded_sentence = decode_sequence(encoder_input_data, model, encoder_model, decoder_model).lstrip()

    return decoded_sentence

if __name__ == "__main__":
    model = load_model(m_path + 'elapsed_seq2seq.h5')
    model.load_weights(m_path + 'weights.h5')
    encoder_model = load_model(m_path+'encoder.h5')
    decoder_model = load_model(m_path+'decoder.h5')

    formula = 'exists e1 e2 x1 (_run(e1) & _walk(e2) & (Subj(e1) = x1) & (Subj(e2) = x1) & _boy(x1))'
    test = surface_realization(formula, model, encoder_model, decoder_model)
    print(test)
