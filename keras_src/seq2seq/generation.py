# coding: utf-8
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model

import txt_tool
import corpus
from decoder import decode_sequence, decode_sequence_with_mask
from masking import get_masking_vector

m_path = corpus.m_path

def surface_realization(formula, model, encoder_model, decoder_model):
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
    model = load_model(m_path + 'elapsed_seq2seq.h5')
    #m = load_model(m_path + 'elapsed_seq2seq.h5') #kesu
    #m.save_weights(m_path + 'weights.h5')#kesu
    model.load_weights(m_path + 'weights.h5')
    encoder_model = load_model(m_path+'encoder.h5')
    decoder_model = load_model(m_path+'decoder.h5')
    #formula='exists x.(_dog(x) & exists z00.(_field(z00) & exists e.(_in(e,z00) & (Subj(e) = x))))'
    formula = 'exists e1 e2 x1 (_run(e1) & _walk(e2) & (Subj(e1) = x1) & (Subj(e2) = x1) & _boy(x1))'
    test = surface_realization(formula, model, encoder_model, decoder_model)
    print("\n\nresult...\n\n")
    print("formula: ", formula)
    print("\n\n")
    print("decoded: ", test)
    print("\n\n...\n\n")
