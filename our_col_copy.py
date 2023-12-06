from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
import torch
import tqdm
from nltk.corpus import stopwords
from models.bert_models import BertForLM, BertForConcatNextSentencePrediction
from pattern3.text.en import singularize, pluralize
import sys
from transformers import BertTokenizer, GPT2Tokenizer
from dataloader import MSMARCODataset
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# nltk.download('stopwords')

COMMON_WORDS = ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it']
STOPWORDS = set(stopwords.words('english'))
NUM_FILTERS = 1000
VERBOSE = True
SEQ_LEN = 30
TOPK = 50
REGULARIZE = False
LR = 0.001
STEMP = 1.0
MAX_ITER = 10
PERTURB_ITER = 30
BETA = 0.0
NUM_BEAMS = 5

def log(msg):
    if msg[-1] != '\n':
        msg += '\n'
    sys.stderr.write(msg)
    sys.stderr.flush()

def add_single_plural(text, tokenizer):
    '''
    returns a list of tokens 
    - from text that are: singularized, pluralized
    - from tokenizer vocab that are related to a word in the text
    '''
    tokens = tokenizer.tokenize(text)
    # stores words from tokenizer that contain or are contained by a word in the query
    contains = []
    for word in tokenizer.vocab:
        if word.isalpha() and len(word) > 2:
            for t in tokens:
                if len(t) > 2 and word != t and (word.startswith(t) or t.startswith(word)):
                    contains.append(word)

    for t in tokens[:]:
        if not t.isalpha():
            continue
        sig_t = singularize(t)
        plu_t = pluralize(t)
        if sig_t != t and sig_t in tokenizer.vocab:
            tokens.append(sig_t)
        if plu_t != t and plu_t in tokenizer.vocab:
            tokens.append(plu_t)

    return [w for w in tokens + contains if w not in STOPWORDS]

def get_inputs_filter_ids(inputs, tokenizer):
    '''
    convert input words to ids
    '''
    tokens = [w for w in tokenizer.tokenize(inputs) if w.isalpha() and w not in STOPWORDS]
    return tokenizer.convert_tokens_to_ids(tokens)

def update_ranking(scores_dict, query, old_score, new_score):
    """
    Update the ranking of a document after its score has changed.

    :param scores_dict: Dictionary with query text as keys and lists of scores as values.
    :param query: The query text corresponding to the scores list.
    :param old_score: The old score of the document.
    :param new_score: The new score of the document.
    :return: The new ranking of the document.
    """

    scores = scores_dict[query]
    try:
        scores.remove(old_score)
    except ValueError:
        raise ValueError("Old score not found in the scores list")
    scores.append(new_score)
    scores.sort(reverse=True)  # Assuming higher scores are better
    new_ranking = scores.index(new_score) + 1

    return new_ranking

def find_filters(query, model, tokenizer, device, k=500):
    '''
    returns list of k words in the tokenizer vocab that have the highest similarity score with the query
    '''
    # 22351
    words = [w for w in tokenizer.vocab if w.isalpha() and w not in STOPWORDS]
    # ? 
    inputs = tokenizer.batch_encode_plus([[query, w] for w in words], pad_to_max_length=True)
    all_input_ids = torch.tensor(inputs['input_ids'], device=device)
    all_token_type_ids = torch.tensor(inputs['token_type_ids'], device=device)
    all_attention_masks = torch.tensor(inputs['attention_mask'], device=device)
    n = len(words)
    batch_size = 512
    n_batches = n // batch_size + 1
    all_scores = []
    for i in tqdm.trange(n_batches, desc='Filtering vocab'):
        # work with 512 tokenized query-word pairs per iteration
        # all three shape: 512x6
        input_ids = all_input_ids[i * batch_size: (i + 1) * batch_size]
        token_type_ids = all_token_type_ids[i * batch_size: (i + 1) * batch_size]
        attention_masks = all_attention_masks[i * batch_size: (i + 1) * batch_size]
        outputs = model.forward(input_ids, attention_masks, token_type_ids)
        # (512x1)
        scores = outputs[0][:, 1]
        all_scores.append(scores)
    # (22351x1)
    all_scores = torch.cat(all_scores)
    _, top_indices = torch.topk(all_scores, k)
    # set of words that have the highest score
    filters = set([words[i.item()] for i in top_indices])
    return [w for w in filters if w.isalpha()]

def get_sub_masks(tokenizer, device, prob=False):
    # masking for all subwords in the vocabulary
    vocab = tokenizer.get_vocab()

    def is_special_token(w):
        if isinstance(tokenizer, BertTokenizer) and w.startswith('##'):
            return True
        if isinstance(tokenizer, GPT2Tokenizer) and not w.startswith('Ġ'):
            return True
        if w[0] == '[' and w[-1] == ']':
            return True
        if w[0] == '<' and w[-1] == '>':
            return True
        if w in ['=', '@', 'Ġ=', 'Ġ@'] and w in vocab:
            return True
        return False
    # get token ids for words that are special tokens
    filter_ids = [vocab[w] for w in vocab if is_special_token(w)]
    if prob:
        prob_mask = torch.ones(tokenizer.vocab_size, device=device)
        prob_mask[filter_ids] = 0.
    else:
        prob_mask = torch.zeros(tokenizer.vocab_size, device=device)
        prob_mask[filter_ids] = -1e9
    return prob_mask

def create_constraints(seq_len, tokenizer, device, prob=False):
    stopword_ids = [tokenizer.vocab[w] for w in COMMON_WORDS[:5] if w in tokenizer.vocab]
    if prob:
        masks = torch.zeros(seq_len, tokenizer.vocab_size, device=device)
    else:
        # 16x30522
        masks = torch.zeros(seq_len, tokenizer.vocab_size, device=device) - 1e9

    for t in range(seq_len):
        # for second half of rows 
        if t >= seq_len // 2:
            # set value of common words to 0
            masks[t, stopword_ids] = 1.0 if prob else 0.0
        else:
            # for first half of rows, set full row to 0
            masks[t] = 1.0 if prob else 0.

    return masks

def tokenize_adversarial_example(input_ids, tokenizer):
    if not isinstance(input_ids, list):
        input_ids = input_ids.squeeze().cpu().tolist()

    # make sure decoded string can be tokenized to the same tokens
    sep_indices = []
    for i, token_id in enumerate(input_ids):
        if token_id == tokenizer.sep_token_id:
            sep_indices.append(i)

    if len(sep_indices) == 1 or tokenizer.sep_token_id == tokenizer.cls_token_id:
        # input is a single text
        decoded = tokenizer.decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        encoded_ids = tokenizer.encode(decoded)
    else:
        # input is a pair of texts
        assert len(sep_indices) == 2, sep_indices
        a_input_ids = input_ids[1:sep_indices[0]]
        b_input_ids = input_ids[sep_indices[0] + 1: sep_indices[1]]
        a_decoded = tokenizer.decode(a_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        b_decoded = tokenizer.decode(b_input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        encoded_ids = tokenizer.encode(a_decoded, b_decoded)

    return encoded_ids

def valid_tokenization(input_ids, tokenizer: BertTokenizer, verbose=False):
    '''
    check if de- and encoding the collision leads to that same collision
    '''
    input_ids = input_ids.squeeze().cpu().tolist()
    # add cls and sep tokens to collision
    if input_ids[0] != tokenizer.cls_token_id:
        input_ids = [tokenizer.cls_token_id] + input_ids
    if input_ids[-1] != tokenizer.sep_token_id:
        input_ids = input_ids + [tokenizer.sep_token_id]

    # make sure decoded string can be tokenized to the same tokens
    encoded_ids = tokenize_adversarial_example(input_ids, tokenizer)
    valid = len(input_ids) == len(encoded_ids) and all(i == j for i, j in zip(input_ids, encoded_ids))
    if verbose and not valid:
        log(f'Inputs: {tokenizer.convert_ids_to_tokens(input_ids)}')
        log(f'Re-encoded: {tokenizer.convert_ids_to_tokens(encoded_ids)}')
    return valid, encoded_ids

def gen_aggressive_collision(inputs_a, inputs_b, model, tokenizer, device, margin=None, lm_model=None):
    # get word embedding layer parameters (30522x1024)
    word_embedding = model.get_input_embeddings().weight.detach()

    if lm_model is not None:
        # same for LM embedding layer (30522x768)
        lm_word_embedding = lm_model.get_input_embeddings().weight.detach()
    # 30522
    vocab_size = word_embedding.size(0)
    # (30522x1)
    input_mask = torch.zeros(vocab_size, device=device)
    # list of k words from tokenizer vocab that are very similar to query
    filters = find_filters(inputs_a, model, tokenizer, device, k=NUM_FILTERS)
    # list of ids for the words in the best matching sentences concat
    best_ids = get_inputs_filter_ids(inputs_b, tokenizer)
    # model vocab size --> change vals of words from best sentences concat
    input_mask[best_ids] = -1e9
    # list of enrichted query tokens (singular, plural, similar root)
    remove_tokens = add_single_plural(inputs_a, tokenizer)
    # if VERBOSE:
    #     log(','.join(remove_tokens))
    # list of token ids according to tokenizer of enriched query tokens
    remove_ids = tokenizer.convert_tokens_to_ids(remove_tokens)
    # add token id of .
    remove_ids.append(tokenizer.vocab['.'])
    # also change vals of enriched query tokens
    input_mask[remove_ids] = -1e9
    num_filters_ids = tokenizer.convert_tokens_to_ids(filters)
    # also change vals of vocab words that are similar to query according to model
    input_mask[num_filters_ids] = -1e9
    # array of size <tokenizer vocab>: everything 0, except special tokens are -1e9
    sub_mask = get_sub_masks(tokenizer, device)

    input_ids = tokenizer.encode(inputs_a)
    # array of encoding numbers (query length + 2)
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    # prevent output num_filters neighbor words
    seq_len = SEQ_LEN
    # shape: topK x len(input_ids) [the 0 does nothing]
    batch_input_ids = torch.cat([input_ids] * TOPK, 0)
    # mask of shape 16xvocab size
    stopwords_mask = create_constraints(seq_len, tokenizer, device)

    def relaxed_to_word_embs(x):
        # convert relaxed inputs to word embedding by softmax attention
        # seqlenxV filled with zeros --> devaluate special tokens and similar vocab words
        masked_x = x + input_mask + sub_mask
        if REGULARIZE:
            masked_x += stopwords_mask
        # p = \hat{c}_t in the paper; prob of each word in vocab
        # STEMP --> 1 means sharper prob assignment
        # (30 x 30522)
        p = torch.softmax(masked_x / STEMP, -1)
        # p * word_embedding; each row is the same (30 x 1024)
        x = torch.mm(p, word_embedding)
        # add embeddings for period and SEP
        # (31 x 1024)
        x = torch.cat([x, word_embedding[tokenizer.sep_token_id].unsqueeze(0)])
        return p, x.unsqueeze(0)

    def get_lm_loss(p):
        x = torch.mm(p.detach(), lm_word_embedding).unsqueeze(0)
        return lm_model(inputs_embeds=x, one_hot_labels=p.unsqueeze(0))[0]

    # shape: topKx1 (filled with the id of the separator token)
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * TOPK, device=device)
    # shape: topK x 1024 [selects row #sep_token_id --> stacks topK times]
    batch_sep_embeds = word_embedding[sep_tensor].unsqueeze(1)
    # tensor([1])
    labels = torch.ones((1,), dtype=torch.long, device=device)
    repetition_penalty = 1.0

    best_collision = None
    best_score = -1e9
    prev_score = -1e9
    collision_cands = []

    # (30, 30522)
    var_size = (seq_len, vocab_size)
    # 30x30522 filled with zeros; THIS IS WHAT GETS OPTIMIZED!
    z_i = torch.zeros(*var_size, requires_grad=True, device=device)
    # 20
    for it in tqdm.tqdm(range(MAX_ITER)):
        # 0.001; optimizes z_i!
        optimizer = torch.optim.Adam([z_i], lr=LR)
        # 30
        for j in range(PERTURB_ITER):
            optimizer.zero_grad()
            # relaxation
            # (30x30522) and (31x1024)
            p_inputs, inputs_embeds = relaxed_to_word_embs(z_i)
            # forward to BERT with relaxed inputs to calculate S(x,c)
            # loss: 1 number; cls_logits: 2 numbers
            loss, cls_logits, _ = model(input_ids, inputs_embeds=inputs_embeds, next_sentence_label=labels)
            if margin is not None:
                # add to loss: difference between best BERT score and prediction if BERT score is better, else 0
                loss = loss + torch.sum(torch.relu(margin - cls_logits[:, 1]))

            if BETA > 0.:
                lm_loss = get_lm_loss(p_inputs)
                loss = BETA * lm_loss + (1 - BETA) * loss

            loss.backward()
            optimizer.step()
            if VERBOSE and (j + 1) % 10 == 0:
                log(f'It{it}-{j + 1}, loss={loss.item()}')

        # detach to free GPU memory
        # 30x30522
        z_i = z_i.detach()

        # returns topk word indeces from vocab: (30x50)
        _, topk_tokens = torch.topk(z_i, TOPK)
        # transforms word logits to probs --> expands to top5 copies: (50x30x30522)
        probs_i = torch.softmax(z_i / STEMP, -1).unsqueeze(0).expand(TOPK, seq_len, vocab_size)

        output_so_far = None
        # beam search left to right
        for t in range(seq_len):
            # select one row (corresponds to one collision word): (1x50)
            t_topk_tokens = topk_tokens[t]
            # (50x30522): each row corresponds to a topk word, and for each row the 1
            # indicates this word
            t_topk_onehot = torch.nn.functional.one_hot(t_topk_tokens, vocab_size).float()
            next_clf_scores = []
            for j in range(NUM_BEAMS):
                next_beam_scores = torch.zeros(tokenizer.vocab_size, device=device) - 1e9
                if output_so_far is None:
                    context = probs_i.clone()
                else:
                    output_len = output_so_far.shape[1]
                    beam_topk_output = output_so_far[j].unsqueeze(0).expand(TOPK, output_len)
                    beam_topk_output = torch.nn.functional.one_hot(beam_topk_output, vocab_size)
                    context = torch.cat([beam_topk_output.float(), probs_i[:, output_len:].clone()], 1)
                context[:, t] = t_topk_onehot
                context_embeds = torch.einsum('blv,vh->blh', context, word_embedding)
                context_embeds = torch.cat([context_embeds, batch_sep_embeds], 1)
                clf_logits = model(input_ids=batch_input_ids, inputs_embeds=context_embeds)[0]
                clf_scores = clf_logits[:, 1].detach().float()
                next_beam_scores.scatter_(0, t_topk_tokens, clf_scores)
                next_clf_scores.append(next_beam_scores.unsqueeze(0))

            next_clf_scores = torch.cat(next_clf_scores, 0)
            next_scores = next_clf_scores + input_mask + sub_mask

            if REGULARIZE:
                next_scores += stopwords_mask[t]

            if output_so_far is None:
                next_scores[1:] = -1e9

            if output_so_far is not None and repetition_penalty > 1.0:
                lm_model.enforce_repetition_penalty_(next_scores, 1, NUM_BEAMS, output_so_far, repetition_penalty)

            # re-organize to group the beam together
            # (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(1, NUM_BEAMS * vocab_size)  # (batch_size, num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(next_scores, NUM_BEAMS, dim=1, largest=True, sorted=True)
            # next batch beam content
            next_sent_beam = []
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens[0], next_scores[0])):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size
                next_sent_beam.append((beam_token_score, token_id, beam_id))

            next_batch_beam = next_sent_beam
            # sanity check / prepare next batch
            assert len(next_batch_beam) == NUM_BEAMS
            beam_tokens = torch.tensor([x[1] for x in next_batch_beam], device=device)
            beam_idx = torch.tensor([x[2] for x in next_batch_beam], device=device)

            # re-order batch
            if output_so_far is None:
                output_so_far = beam_tokens.unsqueeze(1)
            else:
                output_so_far = output_so_far[beam_idx, :]
                output_so_far = torch.cat([output_so_far, beam_tokens.unsqueeze(1)], dim=-1)
        ## end of beam search ##

        # output_so_far: (5[num beams]x30[collision length]) with very similar values, but not identical
        # still the same as at the start
        # batch_input_ids: (50x4)
        # still the same as at the start
        # sep_tensor: (50)

        # (5x31): add separation token to each beam row
        pad_output_so_far = torch.cat([output_so_far, sep_tensor[:NUM_BEAMS].unsqueeze(1)], 1)
        # (5x4) [5 times the input] + (5x31) [5 times the collision] = (5x35)
        concat_input_ids = torch.cat([batch_input_ids[:NUM_BEAMS], pad_output_so_far], 1)
        # (5x4) [zeros] + (5x31) [ones] = (5x35): to distinguish the two sequences
        token_type_ids = torch.cat([torch.zeros_like(batch_input_ids[:NUM_BEAMS]),
                                    torch.ones_like(pad_output_so_far)], 1)
        # (5x2): for each beam collision and query, calculates the score
        clf_logits = model(input_ids=concat_input_ids, token_type_ids=token_type_ids)[0]
        # (5x1): second column of clf_logits
        actual_clf_scores = clf_logits[:, 1]
        # indeces of scores in descending order
        sorter = torch.argsort(actual_clf_scores, -1, descending=True)

        # if VERBOSE:
        #     if margin is not None:
        #         decoded = [
        #             f'{actual_clf_scores[i].item():.4f}, '
        #             f'{tokenizer.decode(output_so_far[i].cpu().tolist())}'
        #             for i in sorter
        #         ]
        #         log(f'It={it}, margin={margin:.4f}, query={inputs_a} | ' + ' | '.join(decoded))

        # collision index with highest score
        valid_idx = sorter[0]
        valid = False
        # if the collision is valid
        for idx in sorter:
            # input: collision (row of 30 values)
            valid, _ = valid_tokenization(output_so_far[idx], tokenizer)
            if valid:
                valid_idx = idx
                break

        # re-initialize z_i
        # best collision (1x30)
        curr_best = output_so_far[valid_idx]
        # one hot version of best collision (30 x 30522)
        next_z_i = torch.nn.functional.one_hot(curr_best, vocab_size).float()
        eps = 0.1
        # label smoothing of one hot representation --> z_i for next iteration
        next_z_i = (next_z_i * (1 - eps)) + (1 - next_z_i) * eps / (vocab_size - 1)
        z_i = torch.nn.Parameter(torch.log(next_z_i), True)
        # score of best collision
        curr_score = actual_clf_scores[valid_idx].item()
        # update best score if this one is better and valid
        if valid and curr_score > best_score:
            best_score = curr_score
            # get collision text
            best_collision = tokenizer.decode(curr_best.cpu().tolist())
        # stop if the collision score does not improve
        if curr_score <= prev_score:
            break
        prev_score = curr_score
    # collision text, collision similarity score with query
    return best_collision, best_score, collision_cands

def give_scores_and_ranks(model, query, data, B=10, K=10):

    # Perform model prediction
    scores = model.predict([(query, line['doc']) for line in data])
    # Pair each score with its corresponding doc_id
    score_docid_pairs = list(zip(scores, [line['doc_id'] for line in data]))

    #***
    score_doctxt_pairs = list(zip(scores, [line['doc'] for line in data]))
    sorted_doctxt_topK = [tup[1] for tup in sorted(score_doctxt_pairs, key=lambda x: x[0], reverse=True)[:K]]
    #***

    # **
    docid_doctxt_pairs = {}
    for line in data:
        docid_doctxt_pairs[line['doc_id']] = line['doc']
    # **

    # Sort the pairs by score in descending order to maintain original ranking
    sorted_pairs = sorted(score_docid_pairs, key=lambda x: x[0], reverse=True)
    # Create dictionary with sorted scores
    query_scores_dict = {query: [score for score, _ in sorted_pairs]}
    # Initialize bottom_k_dict
    bottom_b_dict = {query: {}}
    # Iterate through sorted_pairs and add only bottom K documents to bottom_k_dict
    for rank, (score, doc_id) in enumerate(sorted_pairs, start=1):
        if rank > len(sorted_pairs) - B:
            bottom_b_dict[query][doc_id] = (rank, score)

    return bottom_b_dict, query_scores_dict, sorted_doctxt_topK, docid_doctxt_pairs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    model = BertForConcatNextSentencePrediction.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
    model.to(device)
    print('DEVICE:', device)
    print('Loading data...')
    dataset = MSMARCODataset('top1000.dev')
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    lm_model = BertForLM.from_pretrained('collision/collision/wiki103/bert')
    lm_model.to(device)
    lm_model.eval()

    MYMODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2', max_length=512)

    for (query_id, query_text) in dataset.get_query_id_tuples():
        data_for_query = dataset.get_data_for_query_id(query_id)
        target_q_doc, query_scores, topk_sentences, docid_doctxt_dict = give_scores_and_ranks(MYMODEL, query_text, data_for_query, 10, 10)
        best_sent = ' '.join(topk_sentences[:]).rstrip()
        # can be left out and set to None
        best_score = None

        # query text, top-K sentences (according to BERT) concated, _, _, _, best BERT score, _
        print('Generating collision of length', SEQ_LEN)
        collision, col_score, _  = gen_aggressive_collision(query_text, 
                                                            best_sent, 
                                                            model, tokenizer, device, best_score, lm_model)
        msg = f'Query={query_text}\n' \
              f'Best Collision={collision}\n' \
              f'Collision/query similarity score={col_score}\n'
        log(msg)

        if VERBOSE:
            log('---Rank shifts for 10 least relevant documents---')
            # go over bottom docs according to Our BERT
            for did in target_q_doc[query_text].keys():
                
                doctxt = docid_doctxt_dict[did]
                new_score = MYMODEL.predict((query_text, doctxt + ' ' + collision))

                # old rank and score from Our BERT (for bottom 10 docs)
                old_rank, old_score = target_q_doc[query_text][did]
                new_rank = update_ranking(query_scores, query_text, old_score, new_score)

                log(f'Query id={query_id}, Doc id={did}, '
                    f'old score={old_score:.2f}, new score={new_score:.2f}, old rank={old_rank}, new rank={new_rank}')
                
        # TEMP
        break
if __name__ == '__main__':
    main()