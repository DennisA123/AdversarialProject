import random
import torch
import numpy as np
import argparse
from sentence_transformers import CrossEncoder
from itertools import islice

# import from other files
from dataloader import MSMARCO_REL
from semantic_collisions import gen_aggressive_collision
from perturb_doc import perturb_doc, find_important_words
from evaluation import evaluation_ndcg, average_ndcg, succes_rate,average_succes_rate, normalized_shift



VERBOSE = True

def update_ranking(scores, old_score, new_score):
    """
    Update the ranking of a document after its score has changed.

    :param scores_dict: Dictionary with query text as keys and lists of scores as values.
    :param query: The query text corresponding to the scores list.
    :param old_score: The old score of the document.
    :param new_score: The new score of the document.

    :return: The new rank of the document.
    """

    try:
        scores.remove(old_score)
    except ValueError:
        raise ValueError("Old score not found in the scores list")
    
    scores.append(new_score)
    scores.sort(reverse=True)
    new_rank = scores.index(new_score) + 1

    return new_rank

def obtain_new_ranking(old_ranking, dct_new_scores):
    """
    Update the ranking of a document after its score has changed.

    :param old_ranking: Dictionary that represents the old ranking (before perturbation) with doc_ids as keys, and relevance scores as values.
    :param dct_new_scores: Dict of all perturbed docs (keys) and their new scores (rank)

    :return: The complete new ranking of the query, dict of document_ids (keys) in order of relevance (relevance score is value).
    """

    # loop through new scores of all perturbed docs
    for item in dct_new_scores.items():
        old_ranking[item[0]] = item[1]
    
    # sort adapted dict with new scores to obtain new ranking
    new_ranking = dict(sorted(old_ranking.items(), key=lambda item: item[1]))

    return new_ranking

def give_scores_and_ranks(model, query_id, dataset, B, K):

    q_text = dataset.qid_qtxt[query_id]
    doc_ids = dataset.qid_did[query_id]
    print('DOC_IDs that are from give_scores_and_ranks', doc_ids)
    print('DOC_IDs that are from give_scores_and_ranks', len(doc_ids))
    list_dtxts = [dataset.did_dtxt[doc_id] for doc_id in doc_ids]
    # Perform model prediction
    scores = model.predict([(q_text, dtxt) for dtxt in list_dtxts])
    print('Size of scores', len(scores))
    # Pair each score with its corresponding doc_id
    score_docid_pairs = list(zip(scores, doc_ids))
    print('Size of score pairs', len(score_docid_pairs))

    for item in score_docid_pairs:
        if item[1] in ['3928996', '1901494', '7457306', '1508194', '4299161']:
            print(item)

    # create sorted dictionary
    # swap keys and values such that we have doc_id: score
    complete_ranking_unswap = dict(sorted(score_docid_pairs, key=lambda x: x[0]))
    print('Complete ranking unswap', len(complete_ranking_unswap))
    complete_ranking = {v: k for k, v in complete_ranking_unswap.items()}
    print('Complete ranking', len(complete_ranking))

    print('---------------------------')
    for item in doc_ids:
        if item not in complete_ranking.keys():
            print('THIS ITEM IS NOT IN COMPLETE RANKING')
            print(item)
    print('--------')

    score_doctxt_pairs = list(zip(scores, list_dtxts))
    sorted_doctxt_topK = [tup[1] for tup in sorted(score_doctxt_pairs, key=lambda x: x[0], reverse=True)[:K]]

    # Sort the pairs by score in descending order to maintain original ranking
    sorted_pairs = sorted(score_docid_pairs, key=lambda x: x[0], reverse=True)
    query_scores = sorted(scores, reverse=True)
    # Initialize bottom_b_dict
    bottom_b_dict = {}
    # Iterate through sorted_pairs and add only bottom K documents to bottom_k_dict
    for rank, (score, doc_id) in enumerate(sorted_pairs, start=1):
        if rank > len(sorted_pairs) - B:
            bottom_b_dict[doc_id] = (rank, score)

    # doc_id: (rank,score), [descending scores], [text, text, ...]
    return complete_ranking, bottom_b_dict, query_scores, sorted_doctxt_topK

def main_encoding(nr_irrelevant_docs, nr_top_docs, nr_words, perturbation_type, choice_of_words):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('DEVICE:', device)
    print('Loading data...')
    dataset = MSMARCO_REL('/home/scur0990/AdversarialProject-1/top1000.dev')

    MYMODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)

    # for storing evaluation values
    ndcg_scores_new = []
    ndcg_scores_old = []
    succes_rates = []

    # tijdelijk kijken of het werkt op meer queries, verander voor uploaden naar git
    test_dct = islice(dataset.qid_qtxt, 3)


    for q_id in test_dct:
        complete_old_ranking, targeted_docs, query_scores, _ = give_scores_and_ranks(MYMODEL, q_id, dataset, nr_irrelevant_docs, nr_top_docs)
        print('COMPLETE OLD RANKING', complete_old_ranking)

        # dict of new_scores
        dct_new_scores = {}

        print('---Rank shifts for 10 least relevant documents---')
        # go over bottom docs according to our model
        for did in targeted_docs.keys():

            # calculate its new score with the collision
            new_score = MYMODEL.predict((dataset.qid_qtxt[q_id], perturb_doc(dataset.did_dtxt[did], dataset.qid_qtxt[q_id], nr_words, perturbation_type, choice_of_words)))
            # old rank and score from our model
            old_rank, old_score = targeted_docs[did]
            new_rank = update_ranking(query_scores, old_score, new_score) 

            # update dict of new_scores
            dct_new_scores[did] = new_score 
                    
            print(f'Query id={q_id}, Doc id={did}, '
                f'old score={old_score:.2f}, new score={new_score:.2f}, old rank={old_rank}, new rank={new_rank}')

            shift = normalized_shift(new_rank, old_rank, len(targeted_docs))
            print(f'The normalized shift is, ', shift)

        # evaluation: get new ranking for query with new scores for all perturbed docs
        print('--Obtain evaluation for new ranking--')
        complete_new_ranking = obtain_new_ranking(complete_old_ranking, dct_new_scores) 
        # ndcg_new = evaluation_ndcg(complete_new_ranking, q_id) 
        # ndcg_old = evaluation_ndcg(complete_old_ranking, q_id)
        # ndcg_scores_new.append(ndcg_new)   
        # ndcg_scores_old.append(ndcg_old)

        # evaluation: succes_rate
        succ_rate = succes_rate(targeted_docs, complete_new_ranking, top_n=100)
        succes_rates.append(succ_rate)
                
        # temporary: only do first query
        # break
    # average_ndcg_new = average_ndcg(ndcg_scores_new)
    # average_ndcg_old = average_ndcg(ndcg_scores_old)
    avg_succes_rate = average_succes_rate(succes_rates)
    # print(f'The old average nDCG score wass {average_ndcg_old:.2f}')
    # print(f'The new average nDCG score is {average_ndcg_new:.2f}')
    # print(f'The difference in nDCG score is {average_ndcg_old - average_ndcg_new}')
    print(f'The succes rate is {avg_succes_rate}')
    

def main_collision(nr_irrelevant_docs, nr_top_docs, nr_words):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForConcatNextSentencePrediction.from_pretrained('collision/collision/birch/models/msmarco_mb')
    # tokenizer = BertTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    # model = BertForConcatNextSentencePrediction.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    model.to(device)
    print('DEVICE:', device)
    print('Loading data...')
    dataset = MSMARCO_REL('/home/scur0990/AdversarialProject-1/top1000.dev')
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    lm_model = BertForLM.from_pretrained('collision/collision/wiki103/bert')
    lm_model.to(device)
    lm_model.eval()

    MYMODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)

    # for storing evaluation values
    ndcg_scores_new = []
    ndcg_scores_old = []
    succes_rates = []

    for q_id in dataset.qid_qtxt.keys():
        complete_old_ranking, targeted_docs, query_scores, topk_sentences = give_scores_and_ranks(MYMODEL, q_id, dataset, nr_irrelevant_docs, nr_top_docs)
        print('Query scores', query_scores)
        best_sent = ' '.join(topk_sentences[:]).rstrip()

        print('Generating collision of length', SEQ_LEN)
        collision, col_score, _  = gen_aggressive_collision(dataset.qid_qtxt[q_id], 
                                                            best_sent, 
                                                            model, tokenizer, device, None, lm_model)
        msg = f'Query={dataset.qid_qtxt[q_id]}\n' \
              f'Best Collision={collision}\n' \
              f'Collision/query similarity score={col_score}\n'
        print(msg)

        print('---Rank shifts for 10 least relevant documents---')
        # go over bottom docs according to our model
        for did in targeted_docs.keys():
            # calculate its new score with the collision
            new_score = MYMODEL.predict((dataset.qid_qtxt[q_id], dataset.did_dtxt[did] + ' ' + collision))
            # old rank and score from our model
            old_rank, old_score = targeted_docs[did]
            new_rank = update_ranking(query_scores, old_score, new_score)

            # update dict of new_scores
            dct_new_scores[did] = new_score 

            print(f'Query id={q_id}, Doc id={did}, '
                f'old score={old_score:.2f}, new score={new_score:.2f}, old rank={old_rank}, new rank={new_rank}')
        
        # evaluation: get new ranking for query with new scores for all perturbed docs
        print('--Obtain nDCG score for new ranking--')
        complete_new_ranking = obtain_new_ranking(complete_old_ranking, dct_new_scores) 
        ndcg_new = evaluation_ndcg(complete_new_ranking, q_id) 
        ndcg_old = evaluation_ndcg(complete_old_ranking, q_id)
        ndcg_scores_new.append(ndcg_new)   
        ndcg_scores_old.append(ndcg_old)   

        # evaluation: succes_rate
        succes_rate = succes_rate(targeted_docs, complete_new_ranking, top_n=100)
        succes_rates.append(succes_rate)   
        
        # temporary: only do first query
        break
    average_ndcg_new = average_ndcg(ndcg_scores_new)
    average_ndcg_old = average_ndcg(ndcg_scores_old)
    avg_succes_rate = average_succes_rate(lst_succes_rate)
    print(f'The old average nDCG score wass {average_ndcg_old:.2f}')
    print(f'The new average nDCG score is {average_ndcg_new:.2f}')
    print(f'The difference in nDCG score is {average_ndcg_old - average_ndcg_new}')
    print(f'The succes rate is {avg_succes_rate}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # command line args for specifying the situation
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--perturbation-method', type=str, default='semantic_collision', choices=['semantic_collision', 'encoding_attack'], help='Which methods is used for document perturbation')
    parser.add_argument('--nr-irrelevant-docs', type=int, default=10, help='How many irrelevant docs will be perturbed and reranked')
    parser.add_argument('--nr-top-docs', type=int, default=10, help='How many top docs are considered relevant')
    parser.add_argument('--nr-words', type=int, default=3, help='How many words are perturbed')
    parser.add_argument('--perturbation-type', type=str, default='del', choices=['base', 'zwsp', 'zwnj', 'zwj', 'rlo', 'bksp', 'del', 'homo', 'homo2'], help='What kind of perturbation is done')
    parser.add_argument('--choice-of-words', type=str, default='important', choices=['random', 'important', 'unimportant'], help='LHow many top docs are considered relevant')
    args = parser.parse_args()

    if args.perturbation_method == 'semantic_collision':
        main_collision(args.nr_irrelevant_docs, args.nr_top_docs, args.nr_words)
    elif args.perturbation_method == 'encoding_attack':
        main_encoding(args.nr_irrelevant_docs, args.nr_top_docs, args.nr_words, args.perturbation_type, args.choice_of_words)