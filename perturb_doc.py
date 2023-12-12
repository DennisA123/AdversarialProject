#!/usr/bin/env python3
#
# perturbations.py
# Nicholas Boucher - December 2022
# Implements Unicode perturbations.
#
from homoglyphs import Homoglyphs
from splade.models.transformer_rep import Splade
from random import randrange
from transformers import AutoTokenizer
import torch
import random 

perturbations = ['base', 'zwsp', 'zwnj', 'zwj', 'rlo', 'bksp', 'del', 'homo', 'homo2']
homoglyphs = Homoglyphs()
zwsp_map = {}

hg2 = {'-':'−','.':'ꓸ','0':'Ο','1':'𝟷','2':'𝟸','3':'𖼻','4':'４','5':'５','6':'Ⳓ','7':'７','8':'𐌚','9':'Ꝯ','A':'Ꭺ','B':'Β','C':'𐊢','D':'Ꭰ','E':'Ꭼ','F':'𐊇','G':'Ꮐ','H':'Η','I':'Ⅰ','J':'Ꭻ','K':'K','L':'𐐛','M':'Μ','N':'ꓠ','O':'೦','P':'Р','Q':'Ｑ','R':'𖼵','S':'Տ','T':'Ꭲ','U':'Ս','V':'ꛟ','W':'Ԝ','X':'ⵝ','Y':'Ⲩ','Z':'Ꮓ','a':'а','b':'ᖯ','c':'ϲ','d':'ⅾ','e':'е','f':'𝖿','g':'ց','h':'𝗁','i':'𝚒','j':'ј','k':'𝚔','l':'ⅼ','m':'ｍ','n':'ո','o':'𐓪','p':'р','q':'ԛ','r':'𝗋','s':'ꮪ','t':'𝗍','u':'𝗎','v':'∨','w':'ꮃ','x':'᙮','y':'𝗒','z':'ᴢ'}
hg2_rev = {v:k for k,v in hg2.items()}

def perturb(input: str, perturbation: str) -> str:
    """
    Perturb a given string input

    :param input: text that needs to be perturbed.
    :param perturbation: The type of perturbation that needs to be done.

    :return: Perturbed text.
    """

    # unperturbed
    if perturbation == 'base':
        return input
    # invisible character: zero-width-space between all adjacent characters
    elif perturbation == 'zwsp':
        return '\u200B'.join(list(input))
    # invisible character: zero-width-non-joiner between all adjacent characters
    elif perturbation == 'zwnj':
        return '\u200C'.join(list(input))
    # invisble character: zero-width-joiner between all adjacent characters
    elif perturbation == 'zwj':
        return '\u200D'.join(list(input))
    # reordering: wraps text with right-to-left-override and reverses logical order
    elif perturbation == 'rlo':
        return '\u2066\u202E' + input[::-1] + '\u202C\u2069'
    # deletion: inject x followed by u+8 backspace character
    elif perturbation == 'bksp':
        return input[:int(len(input)/2)] + 'X\u0008' + input[int(len(input)/2):]
    # deletion: inject x followed by u+7F backspace character
    elif perturbation == 'del':
        return input[:int(len(input)/2)] + 'X\u007F' + input[int(len(input)/2):]
    # homoglyph: substitute each character with random chosen homoglyph
    elif perturbation == 'homo':
        return ''.join(map(lambda c: (n := homoglyphs.get_combinations(c))[p if (p := n.index(c)+1) < len(n) else 0], list(input)))
    # homoglyphs: homoglyph manually selected to minimize virtual perturbation artifacts rather than random 
    elif perturbation == 'homo2':
        return ''.join(map(lambda c: hg2[c] if c in hg2 else c, list(input)))
    else:
        raise NameError(f'Perturbation {perturbation} not found.')
    

def perturb_doc(doc: dict, query: str, k: int, perturbation: str, words: str) -> dict:
    """
    Perturb an input document

    :param doc: The document that needs to be perturbed.
    :param query: The query to which the document belongs (necessary to find important words).
    :param k: the number of words that need to be perturbed in document.
    :param perturbation: the type of perturbation that needs to be done.
    :param words: the type of words that will be chosen for perturbation (important, unimportant or random)

    :return: Perturbed document.
    """
    doc_lst = doc.split(' ')

    # obtain random words to perturb
    if words == 'random':
        per_words = random.sample(doc_lst, k)

    # obtain important words to perturb
    elif words == 'important':
        per_words = find_important_words(doc, query, k)
    
    # obtain unimportant words to perturb
    elif words == 'unimportant':
        per_words = find_unimportant_words(doc, query, k)
    else:
        raise NameError(f'Insert a correct option for choice of words.')

    # perturb words in document
    for word in per_words:
        idx = doc_lst.index(word)
        perturbed_word = perturb(word, perturbation)
        doc_lst[idx] = perturbed_word
    
    perturbed_doc = " ".join(doc_lst)
    return perturbed_doc


def find_importance_scores(doc, query):
    """
    Function that finds importance/relevance scores of all tokens of doc

    :param doc: The document that needs to be perturbed.
    :param query: The query to which the document belongs.

    :return: the relevance scores per token of the document w.r.t. the doc + query.
    """

    sparse_model_id = 'naver/splade-cocondenser-ensembledistil'

    sparse_model = Splade(sparse_model_id, agg='max')
    sparse_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(sparse_model_id)

    input_str = doc + ' [SEP] ' + query

    # now compute the document representation
    with torch.no_grad():
        doc_rep = sparse_model(d_kwargs=tokenizer(input_str, return_tensors="pt"))["d_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)

        reverse_voc = {token: w for w, token in tokenizer.vocab.items()}

        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        d = {token: weight for token, weight in zip(col, weights)}
        
        sorted_d = {token: value for token, value in sorted(d.items(), key=lambda item: item[1], reverse=True)}
        bow_rep = []
        for token, value in sorted_d.items():
            word = reverse_voc[token]
            if word in doc.split(' '):
                bow_rep.append((word, round(value, 2)))
        # words = list(i[0] for i in bow_rep[:k])
        # print(words)
        return bow_rep


def find_important_words(doc, query, k):
    """
    Function that finds most important words

    :param doc: The document that needs to be perturbed.
    :param query: The query to which the document belongs.
    :param k: number of words chosen

    :return: the k most important words
    """

    word_tuples = find_importance_scores(doc, query)
    words = list(i[0] for i in word_tuples[:k])
    return words


def find_unimportant_words(doc, query, k):
    """
    Function that finds least important words

    :param doc: The document that needs to be perturbed.
    :param query: The query to which the document belongs.
    :param k: number of words chosen

    :return: the k least important scores
    """

    word_tuples = find_importance_scores(doc, query)
    words = list(i[0] for i in word_tuples[-k:])
    return words