import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")

model = AutoModelForSequenceClassification.from_pretrained("amberoad/bert-multilingual-passage-reranking-msmarco")

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

dataset = MSMARCODataset('top1000.dev')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# there are 6668967 query-passage pairs. 
# there are 6980 unique queries.

# Select a random query
unique_queries = set([data['query_id'] for data in dataset.data])
selected_query_id = random.choice(list(unique_queries))

# Retrieve all passages for the selected query
selected_query_passages = [data for data in dataset.data if data['query_id'] == selected_query_id]
selected_query_passages = selected_query_passages[:10]

# Prepare data for the model
inputs = tokenizer(
    [selected_query_passages[0]['query']] * len(selected_query_passages), 
    [p['doc'] for p in selected_query_passages], 
    padding=True, 
    truncation=True, 
    return_tensors='pt',
    max_length=512
)

# Rank the passages using the model
with torch.no_grad():
    model.eval()
    outputs = model(**inputs)
    scores = outputs.logits[:,1]  # Assuming class 1 corresponds to relevant

# Sort the passages based on scores
ranked_passages = sorted(zip(selected_query_passages, scores), key=lambda x: x[1], reverse=True)

# Extract the ranked passages
ranked_passages = [p[0] for p in ranked_passages]

# Now you have your ranked passages for the selected query
