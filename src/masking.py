import corpus
import numpy as np
#input formulas that start "_" means predicate.
#we stemming predicate and put 1 in the dictionary.

def get_masking_vector(inp, verbose=True):
    mask_vec = np.zeros((1, corpus.MAX_DECODER_SEQ_LENGTH, corpus.NUM_DECODER_TOKENS),dtype='float32')

    for num, pred in enumerate(inp) :
        if(pred[:1]=='_'):
            for p in pred.split('_')[1:]:
                try:
                    for word in corpus.dict_lem[p]:
                        index = corpus.target_token_index[word]
                        for i in range(corpus.MAX_DECODER_SEQ_LENGTH):
                            mask_vec[0,i,index] = 1.
                except:
                    if verbose: print("ERROR word : ",p)
                    continue

    for f in corpus.func_index:
        for i in range(corpus.MAX_DECODER_SEQ_LENGTH):
            mask_vec[0,i,f] = 1.
    return mask_vec
