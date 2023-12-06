from sentence_transformers import CrossEncoder

# model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
# tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')

# print(model.bert.embeddings.word_embeddings.weight.detach())

# features = tokenizer('How many people live in Berlin?', 'Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.',  padding=True, truncation=True, return_tensors="pt")

# model.eval()
# with torch.no_grad():
#     scores = model(**features).logits
#     print(scores)

