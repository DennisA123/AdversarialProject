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

# def perturb_word(doc, k, perturbation):
#     doc_lst = doc.split()
#     print(doc)

#     # Load the BERT model and create a new tokenizer 
#     model = transformers.BertModel.from_pretrained("bert-base-uncased") 
#     tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased") 

#     # Tokenize and encode the text 
#     input_ids = tokenizer.encode(doc, add_special_tokens=True) 

#     # Use BERT to encode the meaning and context of the words and phrases in the text 
#     outputs = model(torch.tensor([input_ids])) 

#     # Use the attention weights of the tokens to identify the least important words and phrases 
#     attention_weights = outputs[-1] 
#     irr_tokens = sorted(attention_weights[0], key=lambda x: x[1], reverse=True)[-k:] 

#     # Decode the top tokens and print the top 3 keywords 
#     irr_words = [tokenizer.decode([token[0]]) for token in irr_tokens]
#     print('Irrelevant words:', irr_words)

#     for word in irr_words:
#         idx = doc_lst.index(word)
#         perturbed_word = perturb(irr_words, perturbation)
#         doc_lst[idx] = perturbed_word
#         print('Perturbed word', perturbed_word)
#     perturbed_doc = " ".join(doc_lst)
#     print('Perturbed document', perturbed_doc)
#     return perturbed_doc



def unperturb(input: str, perturbation: str) -> str:
    if perturbation == 'base':
        return input
    elif perturbation == 'zwsp':
        return input.replace('\u200B', '')
    elif perturbation == 'zwnj':
        return input.replace('\u200C', '')
    elif perturbation == 'zwj':
        return input.replace('\u200D', '')
    elif perturbation == 'rlo':
        return input[2:-2][::-1]
    elif perturbation == 'bksp':
        return input[:int((len(input)-2)/2)] + input[int((len(input)-2)/2)+2:]
    elif perturbation == 'del':
        return input[:int((len(input)-2)/2)] + input[int((len(input)-2)/2)+2:]
    elif perturbation == 'homo':
        return ''.join(map(lambda c: (n := homoglyphs.get_combinations(c))[max(n.index(c)-1, 0)], list(input)))
    elif perturbation == 'homo2':
        return ''.join(map(lambda c: hg2_rev[c] if c in hg2_rev else c, list(input)))
    else:
        raise NameError(f'Perturbation {perturbation} not found.')
    

def perturb_doc(doc: dict, query: str, k: int, perturbation: str, words: str) -> dict:
    doc_lst = doc.split(' ')
    print(doc_lst)

    # obtain random words to perturb
    if words == 'random':
        per_words = random.sample(doc_lst, k)

    # obtain important words to perturb
    elif words == 'important':
        per_words = find_important_words(doc, query, k)
    
    # obtain unimportant words to perturb
    elif words == 'unimportant':
        per_words = 0
    else:
        raise NameError(f'Insert a correct option for choice of words.')

    # perturb words in document
    for word in per_words:
        idx = doc_lst.index(word)
        perturbed_word = perturb(word, perturbation)
        doc_lst[idx] = perturbed_word
        # print('Perturbed word: ', perturbed_word)
    
    perturbed_doc = " ".join(doc_lst)
    print('Perturbed document', perturbed_doc)
    return perturbed_doc

def find_important_words(doc, query, k):

    sparse_model_id = 'naver/splade-cocondenser-ensembledistil'

    sparse_model = Splade(sparse_model_id, agg='max')
    sparse_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(sparse_model_id)


    # now compute the document representation
    with torch.no_grad():
        doc_rep = sparse_model(d_kwargs=tokenizer(doc, return_tensors="pt"))["d_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)

        reverse_voc = {v: k for k, v in tokenizer.vocab.items()}
        print(reverse_voc)

        # get the number of non-zero dimensions in the rep:
        col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
        print("number of actual dimensions: ", len(col))

        # now let's inspect the bow representation:
        weights = doc_rep[col].cpu().tolist()
        d = {k: v for k, v in zip(col, weights)}
        sorted_d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}
        bow_rep = []
        for k, v in sorted_d.items():
            bow_rep.append((reverse_voc[k], round(v, 2)))
        words = list(i[0] for i in bow_rep[:k])
        print(words)
        return words