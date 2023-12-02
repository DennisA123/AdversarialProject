from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from models.bert_models import BertForLM
import os

# COLLISION_DIR = os.environ.get('COLLISION_DIR', 'collision/')
# assert os.path.exists(COLLISION_DIR), 'Please create a directory for extracting data and models.'

# BERT_LM_MODEL_DIR = os.path.join('collision', 'wiki103', 'bert')
# print(BERT_LM_MODEL_DIR)

def main():
    with torch.no_grad():
        model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
        tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-12-v2')
        device = torch.device('cpu')
        model.eval()
        model.to(device)

        lm_model = BertForLM.from_pretrained('dir of LM')
        lm_model.to(device)
        lm_model.eval()

        collision, new_score, collision_cands = gen_aggressive_collision('how to wash an expensive cat', 'by putting your cat in your bath and having the water be warm', model, tokenizer, device, 0.9, lm_model)