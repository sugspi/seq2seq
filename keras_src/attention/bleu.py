# coding: utf-8
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model

import txt_tool
import corpus
from decoder import decode_sequence

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

m_path = 'models/'

if __name__ == "__main__":
    model = load_model(m_path + 'elapsed_seq2seq.h5')
    model.load_weights(m_path + 'weights.h5')
    encoder_model = load_model(m_path+'encoder.h5')
    decoder_model = load_model(m_path+'decoder.h5')

    INP_LEN = len(test_seq)
    sum_score = 0
    results = []

    for seq_index in range(INP_LEN):
        test_data, tmp = test_seq[seq_index]
        input_seq = test_data[0]
        #input_mask = test_data[2]
        decoded_sentence = decode_sequence(input_seq).lstrip()
        #results.append(txt_tool.remove_punct(decoded_sentence).split(' '))

        ###############################################################
        #   print files
        ###############################################################
        #fname = 'name.txt'
        #f = open(fname, 'w')
        #f.write(corpus.all_output_expections[seq_index]+'\n')
        #f.write(decoded_sentence.strip()+'\n')
        #f.close()

        ###############################################################
        #   print screen
        ###############################################################
        #print('Input sentence:', input_texts[seq_index])
        print('Answer sentence:', corpus.all_output_expections [seq_index])
        print('Decoded sentence:', decoded_sentence)
        print('')

        anwer_sentences  = corpus.all_output_expections[:4000]
        #bleu = corpus_bleu([[txt_tool.remove_punct(t).split(' ')] for t in anwer_sentences], results)
        #print('bleu score',bleu)
