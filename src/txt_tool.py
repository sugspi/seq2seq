import re

def formula_to_list(form):
    formula = re.sub('\(', ' ',form)
    formula = re.sub('\)', ' ',formula)
    formula = re.sub(',', ' , ',formula)
    formula = formula + 'EOS'
    formula_list = [i for i in re.split('\s|\.', formula) if i not in [''] ]
    return formula_list

def get_predicate_in_formula_list(inp):
    new_formula = []
    for pred in inp:
        if(pred[:1]=='_'):
            new_formula.append(pred)

    return new_formula


def text_to_list(text):
    target = text.lstrip().lower()
    target = 'BOS ' + target + ' EOS'
    target_list = [i for i in re.split('\s|\.', target) if i not in [''] ]
    return target_list

def remove_punct(text):
    if text.endswith('.'):
        return text[:-1]
    else:
        return text
