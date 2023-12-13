import tqdm
import collections
from torch.utils.data import Dataset

class MSMARCO_REL(Dataset):
    def __init__(self, filepath):
        self.qid_qtxt = {}
        self.qid_did = collections.defaultdict(list)
        self.did_dtxt = {}

        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm.tqdm(file, desc='Loading in data:'):
                q_id, doc_id, query, doc = line.split('\t')
                self.qid_qtxt[q_id] = query
                self.qid_did[q_id].append(doc_id)
                self.did_dtxt[doc_id] = doc

class RELEVANCE(Dataset):
    def __init__(self, filepath):
        self.relevance = []

        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm.tqdm(file, desc='Loading in relevance labels:'):
                q_id, _, d_id, _ = line.split('\t')
                self.relevance.append((q_id, d_id)) 
