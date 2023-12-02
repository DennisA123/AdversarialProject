import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sentence_transformers import CrossEncoder

class MSMARCODataset(Dataset):
    def __init__(self, filepath):
        self.data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                query_id, doc_id, query, doc = line.split('\t')
                self.data.append({'query_id': query_id, 'doc_id': doc_id, 'query': query, 'doc': doc})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_docs_for_query_id(self, query_id):
        return [item for item in self.data if item['query_id'] == query_id][:10]
    
def rank_documents(model, query, documents):
    scores = model.predict([(query, doc['doc']) for doc in documents])
    ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc[0]['doc_id'] for doc in ranked_docs]

# there are 6668967 query-passage pairs. 
# there are 6980 unique queries.    
dataset = MSMARCODataset('top1000.dev')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load the dataset
dataset = MSMARCODataset('top1000.dev')

# Load the model
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)

# Select a random query
random_query = random.choice(dataset.data)

# Get all documents for this query
docs_for_query = dataset.get_docs_for_query_id(random_query['query_id'])

# Rank the documents
ranked_doc_ids = rank_documents(model, random_query['query'], docs_for_query)

# Store the ranking in a dictionary
ranking_dict = {random_query['query_id']: ranked_doc_ids}

# Print or return the ranking_dict
print(ranking_dict)





