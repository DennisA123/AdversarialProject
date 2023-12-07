import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

class MSMARCO_REL(Dataset):
    def __init__(self, filepath):
        self.qid_qtxt = {}
        self.qid_did = defaultdict(list)
        self.did_dtxt = {}
        # self.qid_did_rel = defaultdict(dict())

        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm.tqdm(file):
                q_id, doc_id, query, doc, rel = line.split('\t')
                self.qid_qtxt[q_id] = query
                self.qid_did[q_id].append(doc_id)
                self.did_dtxt[doc_id] = doc
                # self.qid_did_rel[q_id][doc_id] = rel
