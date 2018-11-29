import re

def formula_to_list(formula):
    formula = re.sub('\(', ' ( ',formula)
    formula = re.sub('\)', ' ) ',formula)
    formula = re.split('\s|\.', formula)
    formula_list = [i for i in formula if i not in ['TrueP', ''] ]
    formula_list.append('EOS')
    return formula_list

def text_to_list(text):
    text = text.lstrip().lower()
    text = 'BOS ' + text + ' EOS'
    target_list = [i for i in re.split('\s|\.', text) if i not in [''] ]
    return target_list

def remove_punct(text):
    if text.endswith('.'):
        return text[:-1]
    else:
        return text
