# coding: utf-8
import sys
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model

import txt_tool
import corpus
from decoder import decode_sequence, decode_sequence_with_mask
from masking import get_masking_vector

m_path = corpus.m_path

model = load_model(m_path + 'elapsed_seq2seq.h5')
m = load_model(m_path + 'elapsed_seq2seq.h5')
m.save_weights(m_path + 'weights.h5')#kesu
model.load_weights(m_path + 'weights.h5')
encoder_model = load_model(m_path+'encoder.h5')
decoder_model = load_model(m_path+'decoder.h5')

def surface_realization(formula): # model, encoder_model, decoder_model):
    lst_formula = txt_tool.formula_to_list(formula)
    encoder_input_data = np.zeros((1, corpus.MAX_ENCODER_SEQ_LENGTH),dtype='float32')

    for t, token in enumerate(lst_formula):
        if token in corpus.formula_token_index:
            encoder_input_data[0,t] = corpus.formula_token_index[token]

    if(corpus.model_name == 'attention'):
        decoded_sentence = decode_sequence(encoder_input_data, model, encoder_model, decoder_model).lstrip()

    elif(corpus.model_name == 'masking'):
        mask_vector = get_masking_vector(lst_formula)
        decoded_sentence = decode_sequence_with_mask(encoder_input_data, mask_vector, model, encoder_model, decoder_model).lstrip()

    return decoded_sentence


if __name__ == "__main__":
    formula = 'exists x08.(_street(x08) & _diagonal(x08) & _11(x08) & exists e09.(_own(e09) & (Acc(e09) = x08) & exists x010.(_limited(x010) & _properties(x010) & _redefine(x010) & (Subj(e09) = x010))) & exists x011.(_floor(x011) & _20(x011) & exists e012.(_have(e012) & (Subj(e012) = x08) & (Acc(e012) = x011))))'
    test = surface_realization(formula)#, model, encoder_model, decoder_model)
    print("\n\nresult...\n\n")
    print("formula: ", formula)
    print("\n\n")
    print("decoded: ", test)
    print("\n\n...\n\n")
