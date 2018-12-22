First, you need set your path of directories.

you write in dir_path.json:

model_name = 'attention' | 'masking' | 'graph'
m_path = MODEL + experiment (see below)
data_path = path_to_corpus_data
function_words_list = path_to_corpus_data

example:
{
  "model_name" : "masking",
  "m_path" : "masking/mask_train_model/",
  "data_path" : "masking/mask_train_model/snli_0413_formula_1000.txt",
  "function_words_list" : "func_word.txt"  
}

our expected structure of directories
MODEL(attention|masking|graph)
  |-- experiment (including file.h5, info.json, eval.txt, generated_sentences)

when you train your model you need set parameters in param.path_json

or when you eval or generation, you give argument for system,like
"python eval.py model.h5"
