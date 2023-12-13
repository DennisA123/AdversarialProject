import numpy as np
from sklearn.metrics import ndcg_score

def get_rel_labels(q_id, rel_data, dataset):
    """
    Function that obtains the relevance labels given a query

    :param query_id: The id of the query to which the documents are relevant (or not).

    :return: dict with document_ids as keys, and the values are the relevance label (1 if doc is relevant to query, 0 otherwise).
    """

    rel_labels = {}

    doc_ids = dataset.qid_did[q_id]

    for d_id in doc_ids:
        if (q_id, d_id) in rel_data.relevance:
            rel_labels[d_id] = 1
        else:
            rel_labels[d_id] = 0
    return rel_labels

def evaluation_ndcg(complete_ranking, query_id, rel_data, dataset):
    """
    Function that calculates the ndcg of a ranking

    :param complete_ranking: Dictionary that represents the ranking with doc_ids as keys, and relevance scores as values.
    :param query_id: The id of the query to which the ranking belongs.

    :return: nDCG score of ranking.
    """

    # obtain relevance labels of query id
    relevance_labels = get_rel_labels(query_id, rel_data, dataset)

    # sort the relevance labels such that it has same ordering as complete ranking
    sorted_relevance_labels = dict(sorted(relevance_labels.items(), key=lambda x: list(complete_ranking.keys()).index(x[0])))

    assert sorted_relevance_labels.keys() == complete_ranking.keys(), "The doc_ids of ranking and relevance labels do not match."

    # calculate ndcg score
    ndcg = ndcg_score(np.array([list(sorted_relevance_labels.values())]), np.array([list(complete_ranking.values())]))

    return ndcg

def normalized_shift(new_rank, old_rank, total_n_docs):
    """
    Function that calculates the normalized increase of document after perturbation

    :param new_rank: The new position of document in ranking
    :param old_rank: The old position of document in ranking
    :param total_n_docs: The length of the complete ranking in which document occurs

    :return: the normalized shift in ranking
    """
    increase = old_rank - new_rank
    avg_increase = increase / total_n_docs
    return avg_increase

def success_rate(dct_irr_docs, complete_new_ranking, top_n=100):
    """
    Function that calculates the succes rate of a new ranking list

    :param dct_irr_docs: dict of doc_ids of irrelevant docs (which are the keys) that are perturbed
    :param complete_new_ranking: sorted dict with new ranking

    :return: the succes rate of irrelevant docs that are pushed up into top  n
    """
    # obtain sorted list of doc_ids 
    ranking_list = [item[0] for item in complete_new_ranking.items()]

    # count how many of original irr_docs are in top_n of new 
    count_top = 0
    for doc_id in dct_irr_docs.keys():
        index = ranking_list.index(doc_id)
        if index <= top_n:
            count_top += 1

    # succes rate is percentage of irrelevant docs that is re-ranked in top100
    succes_rate = count_top / len(dct_irr_docs)
    return succes_rate