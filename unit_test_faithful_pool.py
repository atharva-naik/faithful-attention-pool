#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import random
import numpy as np
from model.faithful_attn_pooling import FaithfulAttentionPooling
# seed torch, random and numpy.
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# create random sequence vector embeddings and binary attention masks.
SEQ_LEN = 100
BATCH_SIZE = 32
HIDDEN_SIZE = 768
vec_seq = torch.rand(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
base_attn_mask = (torch.rand(BATCH_SIZE, SEQ_LEN)>0.8).float()
# create faithful pooling layer.
faithful_pool = FaithfulAttentionPooling()
token_scores, exp_pool, all_seq_pool, anti_exp_pool = faithful_pool(vec_seq, base_attn_mask)
# check the shape and value of the all sequence pool, explanation tokens pool & the anti-explanation pool.
print("Explanation Pool:", exp_pool.shape) # this is the anchor.
print("Whole Sequence Pool:", all_seq_pool.shape) # this is the positive example.
print("Anti-Explanation Pool:", anti_exp_pool.shape) # this is the negaive example.
# check the shape of token scores.
print(token_scores.shape)
# print all the values:
print(exp_pool)
print(all_seq_pool)
print(anti_exp_pool)
print(token_scores)