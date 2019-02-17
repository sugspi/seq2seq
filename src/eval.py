# coding: utf-8
import sys
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
from keras.utils import Sequence

import txt_tool
import corpus
from decoder import decode_sequence, decode_sequence_with_mask
import train

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu

m_path = corpus.m_path

def eval_blue(test_seq, model, encoder_model, decoder_model):

    INP_LEN = len(test_seq)
    sum_score = 0
    results = []

    for seq_index in range(INP_LEN):
        #seq_index=100
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
        #fname = corpus.model_name + corpus.m_path + 'results.txt'
        #f = open(fname, 'w')
        #f.write(corpus.all_output_expections[seq_index]+'\n')
        #f.write(decoded_sentence.strip()+'\n')
        #f.close()

        ###############################################################
        #   print screen
        ###############################################################
        #print('Input sentence:', test_seq[seq_index])
        print('Answer sentence:', corpus.all_output_expections[seq_index])
        print('Decoded sentence:', decoded_sentence)
        print('')


    anwer_sentences  = corpus.all_output_expections[:2642]
    bleu = corpus_bleu([[txt_tool.remove_punct(t).split(' ')] for t in anwer_sentences], results)

    ###############################################################
    #   print files
    ###############################################################
    #fname = corpus.model_name + corpus.m_path + 'settings.txt'
    #f = open(fname, 'w')
    #f.write('bleu score'+ bleu+'\n')
    #f.close()

    ###############################################################
    #   print screen
    ###############################################################
    print('bleu score: ',bleu)

    return sum_score

if __name__ == "__main__" :
    if(corpus.model_name == 'attention'):
        # train_seq = train.EncDecSequence(corpus.all_input_formulas[6682:], corpus.all_target_texts[6682:], train.batch_size)
        # val_seq = train.EncDecSequence(corpus.all_input_formulas[2640:6682], corpus.all_target_texts[2640:6682], train.batch_size)
        # test_seq = train.EncDecSequence(corpus.all_input_formulas[:2640], corpus.all_target_texts[:2640], 1)

        train_seq = train.EncDecSequence(corpus.all_input_formulas[5075:], corpus.all_target_texts[5075:], train.batch_size)
        val_seq = train.EncDecSequence(corpus.all_input_formulas[2642:5075], corpus.all_target_texts[2642:5075], train.batch_size)
        test_seq = train.EncDecSequence(corpus.all_input_formulas[:2642], corpus.all_target_texts[:2642], 1)


    elif(corpus.model_name == 'masking'):
        # train_seq = train.EncDecSequence(corpus.all_input_formulas[6682:], corpus.all_target_texts[6682:], train.batch_size, 'masking')
        # val_seq = train.EncDecSequence(corpus.all_input_formulas[2640:6682], corpus.all_target_texts[2640:6682], train.batch_size, 'masking')
        # test_seq = train.EncDecSequence(corpus.all_input_formulas[:2640], corpus.all_target_texts[:2640], 1, 'masking')

        train_seq = train.EncDecSequence(corpus.all_input_formulas[5075:], corpus.all_target_texts[5075:], train.batch_size, 'masking')
        val_seq = train.EncDecSequence(corpus.all_input_formulas[2642:5075], corpus.all_target_texts[2642:5075], train.batch_size, 'masking')
        test_seq = train.EncDecSequence(corpus.all_input_formulas[:2642], corpus.all_target_texts[:2642], 1, 'masking')

    args = sys.argv
    model = load_model(args[1])
    m = load_model(args[1])
    m.save_weights(m_path + 'weights.h5')
    model.load_weights(m_path + 'weights.h5')
    encoder_model = load_model(m_path+'encoder.h5')
    decoder_model = load_model(m_path+'decoder.h5')
    eval_blue(test_seq,model,encoder_model,decoder_model)
