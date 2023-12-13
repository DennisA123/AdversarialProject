import pickle

# Read dictionary pkl file
with open('dataset.pkl', 'rb') as fp:
    data = pickle.load(fp)

vocab_dct = {}
for elem in data:
    doc = elem['doc'].split(' ')
    # print(doc)
    for word in doc:
        if word in vocab_dct:
            vocab_dct[word] += 1
        else:
            vocab_dct[word] = 1
    
with open('vocab.pkl', 'wb') as fp:
    pickle.dump(vocab_dct, fp)
    