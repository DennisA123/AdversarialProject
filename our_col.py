from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tqdm
from nltk.corpus import stopwords
from models.bert_models import BertForLM, BertForConcatNextSentencePrediction
from pattern3.text.en import singularize, pluralize
import sys
from transformers import BertTokenizer, GPT2Tokenizer

# nltk.download('stopwords')

COMMON_WORDS = ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it']
STOPWORDS = set(stopwords.words('english'))
NUM_FILTERS = 500
VERBOSE = True
# ?
# SEQ_LEN = 16
SEQ_LEN = 2
TOPK = 50
REGULARIZE = True
LR = 0.001
STEMP = 1.0
MAX_ITER = 20
PERTURB_ITER = 5
BETA = 0.0
# ? 
# NUM_BEAMS = 10
NUM_BEAMS = 4

def log(msg):
    if msg[-1] != '\n':
        msg += '\n'
    sys.stderr.write(msg)
    sys.stderr.flush()

def add_single_plural(text, tokenizer):
    tokens = tokenizer.tokenize(text)
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
    tokens = [w for w in tokenizer.tokenize(inputs) if w.isalpha() and w not in STOPWORDS]
    return tokenizer.convert_tokens_to_ids(tokens)

def find_filters(query, model, tokenizer, device, k=500):
    # ?
    # k = 30
    words = [w for w in tokenizer.vocab if w.isalpha() and w not in STOPWORDS]
    inputs = tokenizer.batch_encode_plus([[query, w] for w in words], pad_to_max_length=True)
    all_input_ids = torch.tensor(inputs['input_ids'], device=device)
    all_token_type_ids = torch.tensor(inputs['token_type_ids'], device=device)
    all_attention_masks = torch.tensor(inputs['attention_mask'], device=device)
    n = len(words)
    batch_size = 512
    n_batches = n // batch_size + 1
    # ?
    n_batches = 1
    all_scores = []
    for i in tqdm.trange(n_batches, desc='Filtering vocab'):
        input_ids = all_input_ids[i * batch_size: (i + 1) * batch_size]
        token_type_ids = all_token_type_ids[i * batch_size: (i + 1) * batch_size]
        attention_masks = all_attention_masks[i * batch_size: (i + 1) * batch_size]
        outputs = model.forward(input_ids, attention_masks, token_type_ids)
        # scores = outputs[0][:]
        # ?
        scores = outputs[0][:, 1]
        all_scores.append(scores)
    all_scores = torch.cat(all_scores)
    # ? dim
    _, top_indices = torch.topk(all_scores, k)
    filters = set([words[i.item()] for i in top_indices])
    # lijst met woorden die iets te maken hebben met query/best sent?
    return [w for w in filters if w.isalpha()]

def get_sub_masks(tokenizer, device, prob=False):
    # masking for all subwords in the vocabulary
    vocab = tokenizer.get_vocab()

    def is_special_token(w):
        # ?
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
        masks = torch.zeros(seq_len, tokenizer.vocab_size, device=device) - 1e9

    for t in range(seq_len):
        if t >= seq_len // 2:
            masks[t, stopword_ids] = 1.0 if prob else 0.0
        else:
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
    input_ids = input_ids.squeeze().cpu().tolist()

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
    word_embedding = model.get_input_embeddings().weight.detach()
    if lm_model is not None:
        lm_word_embedding = lm_model.get_input_embeddings().weight.detach()

    vocab_size = word_embedding.size(0)
    input_mask = torch.zeros(vocab_size, device=device)
    filters = find_filters(inputs_a, model, tokenizer, device, k=NUM_FILTERS)
    best_ids = get_inputs_filter_ids(inputs_b, tokenizer)
    input_mask[best_ids] = -1e9
    remove_tokens = add_single_plural(inputs_a, tokenizer)
    if VERBOSE:
        log(','.join(remove_tokens))

    remove_ids = tokenizer.convert_tokens_to_ids(remove_tokens)
    remove_ids.append(tokenizer.vocab['.'])
    input_mask[remove_ids] = -1e9
    num_filters_ids = tokenizer.convert_tokens_to_ids(filters)
    input_mask[num_filters_ids] = -1e9
    sub_mask = get_sub_masks(tokenizer, device)

    input_ids = tokenizer.encode(inputs_a)
    # structure: tensor([[ 101, 2129, 2000, 9378, 2019, 6450, 4937,  102]])
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    # prevent output num_filters neighbor words
    seq_len = SEQ_LEN
    batch_input_ids = torch.cat([input_ids] * TOPK, 0)
    stopwords_mask = create_constraints(seq_len, tokenizer, device)

    def relaxed_to_word_embs(x):
        # convert relaxed inputs to word embedding by softmax attention
        masked_x = x + input_mask + sub_mask
        if REGULARIZE:
            masked_x += stopwords_mask
        p = torch.softmax(masked_x / STEMP, -1)
        x = torch.mm(p, word_embedding)
        # add embeddings for period and SEP
        x = torch.cat([x, word_embedding[tokenizer.sep_token_id].unsqueeze(0)])
        return p, x.unsqueeze(0)

    def get_lm_loss(p):
        x = torch.mm(p.detach(), lm_word_embedding).unsqueeze(0)
        return lm_model(inputs_embeds=x, one_hot_labels=p.unsqueeze(0))[0]

    # some constants
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * TOPK, device=device)
    batch_sep_embeds = word_embedding[sep_tensor].unsqueeze(1)
    # tensor([1])
    labels = torch.ones((1,), dtype=torch.long, device=device)
    repetition_penalty = 1.0

    best_collision = None
    best_score = -1e9
    prev_score = -1e9
    collision_cands = []

    var_size = (seq_len, vocab_size)
    z_i = torch.zeros(*var_size, requires_grad=True, device=device)
    for it in range(MAX_ITER):
        optimizer = torch.optim.Adam([z_i], lr=LR)
        for j in range(PERTURB_ITER):
            optimizer.zero_grad()
            # relaxation
            # input_embeds shape: (1x17x384) with number values
            p_inputs, inputs_embeds = relaxed_to_word_embs(z_i)
            # forward to BERT with relaxed inputs
            loss, cls_logits, _ = model(input_ids, inputs_embeds=inputs_embeds, next_sentence_label=labels)
            if margin is not None:
                loss += torch.sum(torch.relu(margin - cls_logits[:, 1]))

            if BETA > 0.:
                lm_loss = get_lm_loss(p_inputs)
                loss = BETA * lm_loss + (1 - BETA) * loss

            loss.requires_grad = True
            loss.backward()
            optimizer.step()
            if VERBOSE and (j + 1) % 10 == 0:
                log(f'It{it}-{j + 1}, loss={loss.item()}')

        # detach to free GPU memory
        z_i = z_i.detach()

        _, topk_tokens = torch.topk(z_i, TOPK)
        probs_i = torch.softmax(z_i / STEMP, -1).unsqueeze(0).expand(TOPK, seq_len, vocab_size)

        output_so_far = None
        # beam search left to right
        for t in range(seq_len):
            t_topk_tokens = topk_tokens[t]
            t_topk_onehot = torch.nn.functional.one_hot(t_topk_tokens, vocab_size).float()
            next_clf_scores = []
            for j in range(NUM_BEAMS):
                print(j)
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

        pad_output_so_far = torch.cat([output_so_far, sep_tensor[:NUM_BEAMS].unsqueeze(1)], 1)
        concat_input_ids = torch.cat([batch_input_ids[:NUM_BEAMS], pad_output_so_far], 1)
        token_type_ids = torch.cat([torch.zeros_like(batch_input_ids[:NUM_BEAMS]),
                                    torch.ones_like(pad_output_so_far)], 1)
        clf_logits = model(input_ids=concat_input_ids, token_type_ids=token_type_ids)[0]
        actual_clf_scores = clf_logits[:, 1]
        sorter = torch.argsort(actual_clf_scores, -1, descending=True)
        if VERBOSE:
            decoded = [
                f'{actual_clf_scores[i].item():.4f}, '
                f'{tokenizer.decode(output_so_far[i].cpu().tolist())}'
                for i in sorter
            ]
            log(f'It={it}, margin={margin:.4f}, query={inputs_a} | ' + ' | '.join(decoded))

        valid_idx = sorter[0]
        valid = False
        for idx in sorter:
            valid, _ = valid_tokenization(output_so_far[idx], tokenizer)
            if valid:
                valid_idx = idx
                break

        # re-initialize z_i
        curr_best = output_so_far[valid_idx]
        next_z_i = torch.nn.functional.one_hot(curr_best, vocab_size).float()
        eps = 0.1
        next_z_i = (next_z_i * (1 - eps)) + (1 - next_z_i) * eps / (vocab_size - 1)
        z_i = torch.nn.Parameter(torch.log(next_z_i), True)

        curr_score = actual_clf_scores[valid_idx].item()
        if valid and curr_score > best_score:
            best_score = curr_score
            best_collision = tokenizer.decode(curr_best.cpu().tolist())

        if curr_score <= prev_score:
            break
        prev_score = curr_score

    return best_collision, best_score, collision_cands

def main():
    with torch.no_grad():
        # model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
        # what is the special token for this tokenizer?
        # tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertForConcatNextSentencePrediction.from_pretrained('bert-large-uncased')
        device = torch.device('cpu')
        model.eval()
        model.to(device)

        lm_model = BertForLM.from_pretrained('bert-base-uncased')
        lm_model.to(device)
        lm_model.eval()

        query = 'how to wash an expensive cat'
        best_sent = 'by putting your cat in your bath and having the water be warm'
        best_score = 0.9

        collision, new_score, _  = gen_aggressive_collision(query, 
                                                            best_sent, 
                                                            model, tokenizer, device, best_score, lm_model)
        
        msg = f'Query={query}\n' \
              f'Best true sentences={best_sent}\n' \
              f'Best similarity score={best_score}\n' \
              f'Collision={collision}\n' \
              f'Similarity core={new_score}\n'
        log(msg)

if __name__ == '__main__':
    main()