import transformers 
import torch

# Load the BERT model and create a new tokenizer 
model = transformers.BertModel.from_pretrained("bert-base-uncased") 
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased") 

# Tokenize and encode the text 
input_ids = tokenizer.encode("This is a sample text for keyword extraction.", add_special_tokens=True) 

# Use BERT to encode the meaning and context of the words and phrases in the text 
outputs = model(torch.tensor([input_ids])) 
print(outputs)

# Use the attention weights of the tokens to identify the most important words and phrases 
attention_weights = outputs[-1]
print(model)
print(attention_weights[0].shape)
# sum_attention_weights = attention_weights.sum(dim=-1)
print(attention_weights)
top_tokens = sorted(attention_weights[0], key=lambda x: x[1], reverse=True)[:3] 
print('top tokens', top_tokens)

# Decode the top tokens and print the top 3 keywords 
top_keywords = [tokenizer.decode([token]) for token in top_tokens] 
print('Keywords: ', top_keywords)

























# # # Get attention weights from all layers
# # all_attention_weights = outputs[-2]

# # # Choose the layer you are interested in (e.g., the last layer)
# # layer_attention_weights = all_attention_weights[-1]

# # # Sum the attention weights along the last dimension (token dimension)
# # sum_attention_weights = layer_attention_weights.sum(dim=-1)

# # # Find the indices of the top 3 tokens with the highest attention weights
# # top_indices = sum_attention_weights.argsort(descending=True)[:3]

# # # Decode the top tokens and print the top 3 keywords
# # top_keywords = [tokenizer.decode([index.item()]) for index in top_indices]
# # print(top_keywords)

# # import transformers
# # import torch

# # # Load the BERT model and create a new tokenizer 
# # model = transformers.BertModel.from_pretrained("bert-base-uncased") 
# # tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased") 

# # # Tokenize and encode the text 
# # input_ids = tokenizer.encode("This is a sample text for keyword extraction.", add_special_tokens=True) 

# # # Use BERT to encode the meaning and context of the words and phrases in the text 
# # outputs = model(torch.tensor([input_ids]), output_attentions=True)

# # # Access attention weights from the last layer
# # attention_weights = outputs.attentions[-1]
# # print(attention_weights)

# # # Sum the attention weights along the last dimension (token dimension)
# # sum_attention_weights = attention_weights.sum(dim=-1)
# # print(sum_attention_weights)

# # # Find the indices of the top 3 tokens with the highest attention weights
# # top_indices = sum_attention_weights.argsort(descending=True)[:3]
# # print(top_indices)

# # # Decode the top tokens and print the top 3 keywords
# # top_keywords = [tokenizer.decode(index.item()) for index in top_indices]
# # print(top_keywords)