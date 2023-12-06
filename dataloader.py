import torch
import random
import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sentence_transformers import CrossEncoder

# there are 6668967 query-passage pairs. 
# there are 6980 unique queries.  
class MSMARCODataset(Dataset):
    def __init__(self, filepath):
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in tqdm.tqdm(file):
                query_id, doc_id, query, doc = line.split('\t')
                self.data.append({'query_id': query_id, 'doc_id': doc_id, 'query': query, 'doc': doc})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_data_for_query_id(self, query_id):
        return [item for item in self.data if item['query_id'] == query_id]
    
    def get_query_id_tuples(self):
        return [(item['query_id'], item['query']) for item in self.data]