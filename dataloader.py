import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sentence_transformers import CrossEncoder

N = 1000

class MSMARCODataset(Dataset):
    def __init__(self, filepath):
        self.data = []
        self.doc_data = {}
        self.query_data = {}
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
            # for line in file:
                query_id, doc_id, query, doc = line.split('\t')
                self.data.append({'query_id': query_id, 'doc_id': doc_id, 'query': query, 'doc': doc})
                self.doc_data[doc_id] = doc
                self.query_data[query_id] = query

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
# dataset = MSMARCODataset('top1000.dev')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# print(dataloader)

# Load the dataset
dataset = MSMARCODataset('top1000.dev')
dataset_eval = MSMARCODataset('top1000.eval')

# Load the model
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)
tokenizer = model.tokenizer
tokens = tokenizer.encode("This is a sample text for keyword extraction")
print('Tokens', tokens)
print(tokenizer.decode(tokens))

# Select a random query
random_query = random.choice(dataset.data)
# print(random_query)

# Get all documents for this query
docs_for_query = dataset.get_docs_for_query_id(random_query['query_id'])
# print(docs_for_query)

# Rank the documents
ranked_doc_ids = rank_documents(model, random_query['query'], docs_for_query)

# Store the ranking in a dictionary
ranking_dict = {random_query['query_id']: ranked_doc_ids}

# Print or return the ranking_dict
print(ranking_dict)