#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import random
import numpy as np
from model.bert import FaithfulBertClassifier
from transformers import BertModel, BertTokenizer

# seed torch, random and numpy.
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

bert = BertModel.from_pretrained("bert-base-uncased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tok_args = {
    "padding": "max_length",
    "max_length": 100,
    "truncation": True,
    "return_tensors": "pt"
}
faithful_bert = FaithfulBertClassifier(bert, bert_emb_size=768, num_classes=2,
                                       faithful_head_size=50, faithful_dropout_rate=0.2)

def test_forward_methods(model, *bert_args):
    # test the forward method.
    logits = model(*bert_args)
    print("logits:", logits)
    print("logits.shape:", logits.shape)
    
    # test the faithful forward method.
    token_scores, exp_pool, all_seq_pool, anti_exp_pool, logits = model.faithful_forward(*bert_args)
    print("token_scores:", token_scores)
    print("exp_pool:", exp_pool)
    print("all_seq_pool:", all_seq_pool)
    print("anti_seq_pool:", anti_exp_pool)
    print("logits:", logits)
    print("token_scores.shape:", token_scores.shape)
    print("exp_pool.shape:", exp_pool.shape)
    print("all_seq_pool.shape:", all_seq_pool.shape)
    print("anti_seq_pool.shape:", anti_exp_pool.shape)
    print("logits.shape", logits.shape)

# test forward methods for:
# 1. a single sentence as input.
tok_dict = bert_tokenizer("I like food", **tok_args)
test_forward_methods(
    faithful_bert, tok_dict["input_ids"], 
    tok_dict["attention_mask"],
    tok_dict["token_type_ids"],
)
# 2. a pair of sentences as input.
# 3. a batch of sentences as input.
