import torch
import os
import tqdm
import argparse
from sentence_transformers import CrossEncoder
from itertools import islice

# import from other files
from dataloader import MSMARCO_REL
from models.bert_models import BertForLM, BertForConcatNextSentencePrediction
from transformers import BertTokenizer
from methods.semantic_collisions import gen_aggressive_collision
# ?
# from methods.perturb_doc import perturb_doc
from evaluation import evaluation_ndcg, average_ndcg, succes_rate, average_succes_rate, normalized_shift

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
    list_dtxts = [dataset.did_dtxt[doc_id] for doc_id in doc_ids]
    # Perform model prediction
    scores = model.predict([(q_text, dtxt) for dtxt in list_dtxts])
    # Pair each score with its corresponding doc_id
    score_docid_pairs = list(zip(scores, doc_ids))
    docid_score_pairs = list(zip(doc_ids, scores))

    # create sorted dictionary
    complete_ranking = dict(sorted(docid_score_pairs, key=lambda x: x[0]))


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

def main_encoding(nr_irrelevant_docs, nr_top_docs, nr_words, perturbation_type, choice_of_words, verbosity):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbosity:
        print('DEVICE:', device)
    dataset_path = os.path.join('data', 'top1000.dev')
    dataset = MSMARCO_REL(dataset_path)

    ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)

    # for storing evaluation values
    ndcg_scores_new = []
    ndcg_scores_old = []
    succes_rates = []

    # ?
    test_dct = islice(dataset.qid_qtxt, 3)
    shift_results = []
    for q_id in tqdm.tqdm(test_dct, desc='Going through queries'):
        complete_old_ranking, targeted_docs, query_scores, _ = give_scores_and_ranks(ranker, q_id, dataset, nr_irrelevant_docs, nr_top_docs)

        # dict of new_scores
        dct_new_scores = {}

        if verbosity:
            print('Calculating metrics...')

        # go over bottom docs according to our model
        for did in targeted_docs.keys():

            # calculate its new score with the collision
            new_score = ranker.predict((dataset.qid_qtxt[q_id], perturb_doc(dataset.did_dtxt[did], dataset.qid_qtxt[q_id], nr_words, perturbation_type, choice_of_words)))
            # old rank and score from our model
            old_rank, old_score = targeted_docs[did]
            new_rank = update_ranking(query_scores, old_score, new_score) 

            # update dict of new_scores
            dct_new_scores[did] = new_score 
            
            if verbosity:
                print(f'Query id={q_id}, Doc id={did}, '
                    f'old score={old_score:.2f}, new score={new_score:.2f}, old rank={old_rank}, new rank={new_rank}')

            shift = normalized_shift(new_rank, old_rank, len(targeted_docs))
            shift_results.append(shift)

            if verbosity:
                print('Document with ID',did,'normalized shift:', shift)

        complete_new_ranking = obtain_new_ranking(complete_old_ranking, dct_new_scores) 
        # ndcg_new = evaluation_ndcg(complete_new_ranking, q_id) 
        # ndcg_old = evaluation_ndcg(complete_old_ranking, q_id)
        # ndcg_scores_new.append(ndcg_new)   
        # ndcg_scores_old.append(ndcg_old)   

        # evaluation: succes_rate
        succ_rate = succes_rate(targeted_docs, complete_new_ranking, top_n=100)
        succes_rates.append(succ_rate)   
    
    # average_ndcg_new = average_ndcg(ndcg_scores_new)
    # average_ndcg_old = average_ndcg(ndcg_scores_old)
    avg_succes_rate = average_succes_rate(succes_rates)
    # print(f'The old average nDCG score wass {average_ndcg_old:.2f}')
    # print(f'The new average nDCG score is {average_ndcg_new:.2f}')
    # print(f'The difference in nDCG score is {average_ndcg_old - average_ndcg_new}')

    # Define the directory and file path
    res_directory = './results'
    file_path = f'{res_directory}/results.txt'
    if not os.path.exists(res_directory):
        os.makedirs(res_directory)

    # Write everything to 'results.txt'
    with open(file_path, 'w') as file:
        # Write success rate on the first line, and the shift on second
        file.write(f"{avg_succes_rate}\n")
        avg_shift = sum(shift_results) / len(shift_results)
        file.write(f"{avg_shift}\n")

    print(f'Average success rate:" {avg_succes_rate}')
    print(f'Average normalized shift:" {avg_shift}')

def main_collision(nr_irrelevant_docs, nr_top_docs, nr_words, verbosity, max_iter):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    collision_model_path = os.path.join('models', 'msmarco_mb')
    model = BertForConcatNextSentencePrediction.from_pretrained(collision_model_path)
    # tokenizer = BertTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    # model = BertForConcatNextSentencePrediction.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    model.to(device)
    if verbosity:
        print('DEVICE:', device)
    dataset_path = os.path.join('data', 'top1000.dev')
    dataset = MSMARCO_REL(dataset_path)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    lm_model_path = os.path.join('models', 'bert')
    lm_model = BertForLM.from_pretrained(lm_model_path)
    lm_model.to(device)
    lm_model.eval()

    ranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)

    # for storing evaluation values
    ndcg_scores_new = []
    ndcg_scores_old = []
    succes_rates = []

    # for storing results in the text file
    shift_results = []
    # ?
    test_dct = islice(dataset.qid_qtxt, 2)
    # for q_id in tqdm.tqdm(dataset.qid_qtxt.keys(), desc='Going through queries'):
    for q_id in test_dct:
        complete_old_ranking, targeted_docs, query_scores, topk_sentences = give_scores_and_ranks(ranker, q_id, dataset, nr_irrelevant_docs, nr_top_docs)

        # dict of new_scores
        dct_new_scores = {}

        # print('Query scores', query_scores)
        best_sent = ' '.join(topk_sentences[:]).rstrip()

        if verbosity:
            print('Generating collision of length', nr_words)

        collision, col_score, _  = gen_aggressive_collision(dataset.qid_qtxt[q_id], 
                                                            best_sent, 
                                                            model, tokenizer, device, verbosity, nr_words, max_iter, None, lm_model)
        if verbosity:
            msg = f'Query={dataset.qid_qtxt[q_id]}\n' \
                f'Best Collision={collision}\n' \
                f'Collision/query similarity score={col_score}\n'
            print(msg)

            print('Calculating metrics...')
        # go over bottom docs according to our model
        for did in targeted_docs.keys():
            # calculate its new score with the collision
            new_score = ranker.predict((dataset.qid_qtxt[q_id], dataset.did_dtxt[did] + ' ' + collision))
            # old rank and score from our model
            old_rank, old_score = targeted_docs[did]
            new_rank = update_ranking(query_scores, old_score, new_score)

            # update dict of new_scores
            dct_new_scores[did] = new_score 

            if verbosity:
                print(f'Query id={q_id}, Doc id={did}, '
                    f'old score={old_score:.2f}, new score={new_score:.2f}, old rank={old_rank}, new rank={new_rank}')
                
            shift = normalized_shift(new_rank, old_rank, len(targeted_docs))
            shift_results.append(shift)

            if verbosity:
                print('Document with ID',did,'normalized shift:', shift)
        
        complete_new_ranking = obtain_new_ranking(complete_old_ranking, dct_new_scores) 
        # ndcg_new = evaluation_ndcg(complete_new_ranking, q_id) 
        # ndcg_old = evaluation_ndcg(complete_old_ranking, q_id)
        # ndcg_scores_new.append(ndcg_new)   
        # ndcg_scores_old.append(ndcg_old)   

        # evaluation: succes_rate
        succ_rate = succes_rate(targeted_docs, complete_new_ranking, top_n=100)
        succes_rates.append(succ_rate)   
    
    # average_ndcg_new = average_ndcg(ndcg_scores_new)
    # average_ndcg_old = average_ndcg(ndcg_scores_old)
    avg_succes_rate = average_succes_rate(succes_rates)
    # print(f'The old average nDCG score wass {average_ndcg_old:.2f}')
    # print(f'The new average nDCG score is {average_ndcg_new:.2f}')
    # print(f'The difference in nDCG score is {average_ndcg_old - average_ndcg_new}')


    # Define the directory and file path
    res_directory = './results'
    file_path = f'{res_directory}/results.txt'
    if not os.path.exists(res_directory):
        os.makedirs(res_directory)

    # Write everything to 'results.txt'
    with open(file_path, 'w') as file:
        # Write success rate on the first line, and the shift on second
        file.write(f"{avg_succes_rate}\n")
        avg_shift = sum(shift_results) / len(shift_results)
        file.write(f"{avg_shift}\n")

    print(f'Average success rate: {avg_succes_rate}')
    print(f'Average normalized shift: {avg_shift}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # command line args for specifying the situation
    parser.add_argument('--perturbation_method', type=str, default='semantic_collision', choices=['semantic_collision', 'encoding_attack'], help='Which methods is used for document perturbation')
    parser.add_argument('--nr_irrelevant_docs', type=int, default=10, help='How many irrelevant docs will be perturbed and reranked')
    parser.add_argument('--nr_top-docs', type=int, default=10, help='How many top docs are considered relevant')
    parser.add_argument('--nr_words', type=int, default=3, help='How many words are perturbed')
    parser.add_argument('--perturbation_type', type=str, default='del', choices=['base', 'zwsp', 'zwnj', 'zwj', 'rlo', 'bksp', 'del', 'homo', 'homo2'], help='What kind of perturbation is done')
    parser.add_argument('--choice_of_words', type=str, default='important', choices=['random', 'important', 'unimportant'], help='How many top docs are considered relevant')
    parser.add_argument('--verbosity', action='store_true', help='Print additional information during process')
    parser.add_argument('--max_iter', type=int, default=6, help='How many iterations to find best collision')

    args = parser.parse_args()

    if args.perturbation_method == 'semantic_collision':
        main_collision(args.nr_irrelevant_docs, args.nr_top_docs, args.nr_words, args.verbosity, args.max_iter)
    elif args.perturbation_method == 'encoding_attack':
        main_encoding(args.nr_irrelevant_docs, args.nr_top_docs, args.nr_words, args.perturbation_type, args.choice_of_words, args.verbosity)