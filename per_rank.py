import random
import numpy as np
from per_doc import perturb_doc
from dataloader import MSMARCODataset, rank_documents
from sentence_transformers import CrossEncoder

# Load dataset
dataset = MSMARCODataset('top1000.dev')

# Load model
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)

# Select a random query
random_query = random.choice(dataset.data)
random_query_id = random_query['query_id']

# Get all documents for this query
docs_for_query = dataset.get_docs_for_query_id(random_query_id)

# Rank the documents
ranked_doc_ids = rank_documents(model, random_query_id, docs_for_query)

# Store the ranking in a dictionary
ranking_dict = {random_query['query_id']: ranked_doc_ids}

# Print or return the original ranking_dict
print('Original rank:', ranking_dict)

def perturb_irrelevant_docs(query_id, percentage, original_rank, perturbation, k, words):
    # get number of docs that need to be perturbed
    num_docs = len(original_rank[query_id])
    num_irrelevant_docs = int(np.ceil(percentage * num_docs))
    # print('Number of irr docs: ', num_irrelevant_docs)

    # get most irrelevant docs for perturbation
    irrelevant_doc_ids = original_rank[random_query_id][-num_irrelevant_docs:]
    # print('Irrelevant docs that need to be perturbed: ', irrelevant_doc_ids)
    
    # create new dataset lst with perturbed docs
    dataset_lst = []
    for id in original_rank[random_query_id]:
        if id in irrelevant_doc_ids:
            perturbed_doc = perturb_doc(dataset.query_data[query_id], dataset.doc_data[id], k, perturbation, words)
            dataset_lst.append({'query_id': query_id, 'doc_id': id, 'query': dataset.query_data[query_id], 'doc': perturbed_doc})
        else: 
            dataset_lst.append({'query_id': query_id, 'doc_id': id, 'query': dataset.query_data[query_id], 'doc': dataset.doc_data[id]})
    # print('Len new dataset', len(dataset_lst))

    # re-rank list of docs (with perturbed docs)
    ranked_doc_ids = rank_documents(model, query_id, dataset_lst)
    return {query_id: ranked_doc_ids}


print('New rank: ', perturb_irrelevant_docs(random_query_id, 0.1, ranking_dict, 'homo', 1, 'important'))