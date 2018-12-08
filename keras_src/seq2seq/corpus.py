import re
import txt_tool

model_name = 'masking'
m_path = 'mask_train_model/'
data_path = 'snli_0413_formula_1000.txt'
function_words_list = 'func_word.txt'

###############################################################
#   original corpus data
###############################################################
inp_corpus = open(data_path)
all_input_formulas = []
all_target_texts = []
all_output_expections =[]
#base, surf and func is used only masking model
all_base_texts =[]
all_surf_texts =[]
func_list = []
func_index = []

###############################################################
#   input and output token dictionary
###############################################################
dict_formula_tokens = set()
dict_target_tokens = set()
dict_lem = dict()
formula_token_index = dict()
target_token_index = dict()
reverse_target_token_index = dict()

###############################################################
#   initial demention of word vector
###############################################################
NUM_ENCODER_TOKENS = 0
NUM_DECODER_TOKENS = 0
MAX_ENCODER_SEQ_LENGTH = 0
MAX_DECODER_SEQ_LENGTH = 0

for i, line in enumerate(inp_corpus):
    line = line.split('#') # formula/orign sentence / 動詞とか処理したもの
    inp_formula = line[0]
    target_text = line[1]
    target_text = target_text.lstrip().lower()

    lst_formula = txt_tool.formula_to_list(inp_formula)
    all_input_formulas.append(lst_formula)

    all_output_expections.append(target_text)

    lst_text = txt_tool.text_to_list(target_text)
    all_target_texts.append(lst_text)

    for token in lst_formula:
        if token not in dict_formula_tokens:
            dict_formula_tokens.add(token)

    for token in lst_text:
        if token not in dict_target_tokens:
            dict_target_tokens.add(token)

    ###############################################################
    #   masking model
    ###############################################################

    if(model_name=='masking'):
        base_text = line[2].rstrip().lower()
        base_text = base_text.lstrip()
        base_text = re.split('\s|\.', base_text)
        all_base_texts.append(base_text)

        surf_text = re.split('\s|\.', target_text)
        all_surf_texts.append(surf_text)

        for s,b in (zip(surf_text, base_text)):
            if not(b in dict_lem.keys()):
                dict_lem[b] = set()
            if not(s in dict_lem[b]):
                dict_lem[b].add(s)

    # -------------------- notice: end of inp_corpus --------------------

dict_formula_tokens = sorted(list(dict_formula_tokens))
dict_target_tokens = sorted(list(dict_target_tokens))

NUM_ENCODER_TOKENS = len(dict_formula_tokens) + 1
NUM_DECODER_TOKENS = len(dict_target_tokens) + 1
MAX_ENCODER_SEQ_LENGTH = max([len(txt) for txt in all_input_formulas])
MAX_DECODER_SEQ_LENGTH = max([len(txt) for txt in all_target_texts])

formula_token_index = dict(
    [(token, i+1) for i, token in enumerate(dict_formula_tokens)])
target_token_index = dict(
    [(token, i+1) for i, token in enumerate(dict_target_tokens)])
reverse_target_token_index = dict(
    (i, token) for token, i in target_token_index.items()) #0:tab,1:\n

###############################################################
#   masking model
###############################################################
if(model_name=='masking'):
    f = open(function_words_list)
    line = f.readline()
    while line:
        func_list.append(line.rstrip())
        line = f.readline()
    f.close()

    func_index.append(target_token_index['EOS'])

    for w in func_list:
        if w in target_token_index:
            func_index.append(target_token_index[w])
