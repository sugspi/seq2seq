# coding: utf-8
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model

import txt_tool
import corpus
from decoder import decode_sequence, decode_sequence_with_mask
from masking import get_masking_vector

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

m_path = corpus.m_path

#if __name__ == "__main__":

def eval_blue(test_seq, model, encoder_model, decoder_model):

    INP_LEN = len(test_seq)
    sum_score = 0
    results = []

    for seq_index in range(INP_LEN):
        test_data, tmp = test_seq[seq_index]
        input_seq = test_data[0]

        if(corpus.model_name == 'attention'):
            decoded_sentence = decode_sequence(input_seq, model, encoder_model, decoder_model).lstrip()

        elif(corpus.model_name == 'masking'):
            mask_vector = test_data[2]
            decoded_sentence = decode_sequence_with_mask(input_seq, mask_vector, model, encoder_model, decoder_model).lstrip()

        results.append(txt_tool.remove_punct(decoded_sentence).split(' '))

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

        return sum_score
