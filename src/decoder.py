import corpus
import numpy as np

def decode_sequence(encoder_input_data, model, encoder_model, decoder_model):
    # Encode the input as state vectors.
    e_state, h_state, c_state = encoder_model.predict(encoder_input_data)
    states_value = [h_state,c_state]

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,corpus.MAX_DECODER_SEQ_LENGTH))

    target_seq[0, 0] = corpus.target_token_index['BOS']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq, e_state] + states_value)

        # Sample a token
        sampled_token = ''
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            decoded_sentence += '!'
        else:
            sampled_token = corpus.reverse_target_token_index[sampled_token_index]
            if sampled_token != 'EOS':
                decoded_sentence += sampled_token + ' '

        # Exit condition: either hit max length or find stop token.
        if (sampled_token == 'EOS' or
           len(decoded_sentence) > corpus.MAX_DECODER_SEQ_LENGTH + 15):
            stop_condition = True
            decoded_sentence = decoded_sentence.rstrip()
            decoded_sentence += '.'

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, corpus.MAX_DECODER_SEQ_LENGTH))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

def decode_sequence_with_mask(encoder_input_data, mask_vector, model, encoder_model, decoder_model):
    e_state,h_state,c_state = encoder_model.predict(encoder_input_data)
    states_value = [h_state,c_state]

    target_seq = np.zeros((1,corpus.MAX_DECODER_SEQ_LENGTH))
    target_seq[0, 0] = corpus.target_token_index['BOS']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq, e_state, mask_vector] + states_value)

        sampled_token = ''
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            decoded_sentence += '!'
        else:
            sampled_token =  corpus.reverse_target_token_index[sampled_token_index]
            if sampled_token != 'EOS':
                decoded_sentence += sampled_token + ' '

        # Exit condition: either hit max length or find stop token.
        if (sampled_token == 'EOS' or
           len(decoded_sentence) > corpus.MAX_DECODER_SEQ_LENGTH + 15):
            stop_condition = True
            decoded_sentence = decoded_sentence.rstrip()
            decoded_sentence += '.'

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, corpus.MAX_DECODER_SEQ_LENGTH))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence
