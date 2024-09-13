from nltk.tokenize import RegexpTokenizer
import re
import keyword
from typing import FrozenSet

from utils import Utils

IDENTIFIER_REGEX = re.compile('[_a-zA-Z][_a-zA-Z0-9]*')
string_pattern = r'"([^"\\]*(\\.[^"\\]*)*)"|\'([^\'\\]*(\\.[^\'\\]*)*)\''
code_tokenizer = RegexpTokenizer(r'\w+')

def get_language_keywords(language: str) -> FrozenSet[str]:
    """
    Returns the keywords of a programming language.

    There are some inconsistencies across languages wrt to
    what is considered a keyword. For example, the true/false
    literals are considered keywords in many languages. However,
    we exclude them here for consistency. We also exclude special
    functions-like keywords, such as `die()` in PHP.
    """
    language = language.lower()
    if language == 'python':
        return frozenset(k for k in keyword.kwlist if k != 'True' and k != 'False')
    elif language == 'java':
        f = ['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float', 'for', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while', 'var', 'const', 'goto']
        return frozenset(l.strip() for l in f if len(l.strip()) > 0)

def is_identifier(token, lang=None):
    return True if IDENTIFIER_REGEX.match(token) \
                   and (lang is None or token not in get_language_keywords(lang)) \
        else False

def extract_identifiers(source_code, lang):
    # the main idea is to remove String from a source code
    # then, tokenize the code to get all words and match with identifier regular expression
    # check if it is a language specific keyword, it not, then it is an identifier
    source_code_without_strings = re.sub(string_pattern, '', source_code)
    _ids = [t for t in code_tokenizer.tokenize(source_code_without_strings) if is_identifier(t, lang)]
    return _ids

def compute_id_match(pred_ids, target_ids):
    pred_ids = list(set(pred_ids))
    target_ids = list(set(target_ids))
    tp = 0
    fp = 0
    fn = 0
    for pid in pred_ids:
        if pid in target_ids:
            tp += 1
        else:
            fp += 1
    for tid in target_ids:
        if tid not in pred_ids:
            fn += 1
    return tp, fp, fn

def compute_identifier_match(pred, target, lang):

    comment_prefix = ""
    if lang == "python":
        comment_prefix = "#"
    elif lang == "java":
        comment_prefix = "//"
    
    target_lines = [line.strip() for line in target.splitlines() if line.strip()]
    target_lines = [line for line in target_lines if not line.startswith(comment_prefix)]
    pred_lines = [line.strip() for line in pred.splitlines() if line.strip()]
    pred_lines = [line for line in pred_lines if not line.startswith(comment_prefix)][:len(target_lines)]

    target_str = ''.join(target_lines)
    pred_str = ''.join(pred_lines)

    target_ids = extract_identifiers(target_str, lang)
    pred_ids = extract_identifiers(pred_str, lang)

    id_em = int(target_ids == pred_ids)

    id_tp, id_fp, id_fn = compute_id_match(pred_ids, target_ids)
    id_f1 = 2 * id_tp / (2 * id_tp + id_fp + id_fn) if (2 * id_tp + id_fp + id_fn) != 0 else 0
    return id_em, id_f1


if __name__ == '__main__':
    f_paths = ['predictions/ds7/infile.jsonl', 'predictions/ds7/rg200.jsonl', 'predictions/ds7/rc200.jsonl', 'predictions/ds7/rg-all200.jsonl']
    f_paths = ['predictions/ds7/rg-all200.jsonl', 'predictions/ds7/rg-nodoc200.jsonl', 'predictions/ds7/rg-nosummary200.jsonl']
    for file in f_paths:
        examples = Utils.load_jsonl(file)
        em_sum = 0
        f1_sum = 0
        for example in examples:
            target = example['metadata']['ground_truth']
            pred = example['pred_res']
            em, f1 = compute_identifier_match(pred, target, 'java')
            f1_sum += f1
            em_sum += em
        print(file)
        
        print(format(em_sum/len(examples) * 100, '.2f'))
        print(format(f1_sum/len(examples) * 100, '.2f'))