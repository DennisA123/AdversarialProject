import torch
import random
import tqdm
import collections
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sentence_transformers import CrossEncoder

class MSMARCO_REL(Dataset):
    def __init__(self, filepath):
        self.qid_qtxt = {}
        self.qid_did = collections.defaultdict(list)
        self.did_dtxt = {}
        # self.qid_did_rel = collections.defaultdict(dict())

        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm.tqdm(file):
                # print(line.split('\t'))
                q_id, doc_id, query, doc = line.split('\t')
                self.qid_qtxt[q_id] = query
                self.qid_did[q_id].append(doc_id)
                self.did_dtxt[doc_id] = doc
                # self.qid_did_rel[q_id][doc_id] = rel

class RELEVANCE(Dataset):
    def __init__(self, filepath):
        # self.rel_labels = collections.defaultdict(dict())
        self.relevance = []

        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm.tqdm(file):
                q_id, _, d_id, _ = line.split('\t')
                self.relevance.append((q_id, d_id)) 
                # self.rel_labels[q_id][doc_id] = rel