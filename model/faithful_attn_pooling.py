#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BIG_NUMBER = 999999
# faithful attention pooling layer.
# like single headed attention for now.
# no concept of values here though, only keys and queries are used to calculate the scores.
class FaithfulAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int=768, head_size: int=50, 
                 num_heads: int=1, dropout_rate: float=0.2):
        # layer to extract query embeddings.
        super(FaithfulAttentionPooling, self).__init__()
        self.head_size = head_size
        self.dropout_rate = dropout_rate
        self.query_extractor = nn.Linear(hidden_size, head_size)
        self.key_extractor = nn.Linear(hidden_size, head_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    def get_extended_mask(self, attn_mask):
        """compute the extended attention mask from the binary attention mask."""
        return BIG_NUMBER*(attn_mask[:,None,:]-1) 

    def forward(self, vec_seq: torch.Tensor, 
                base_attn_mask: torch.Tensor,
                percentile_fraction: float=0.2):
        """
        vec_seq: [batch_size x seq_len x hidden_size]
        base_attn_mask: [batch_size x seq_len x hidden_size]
        The base attention mask indicates the point after which pad tokens start.
        """
        # extract the keys & queries.
        keys = self.key_extractor(vec_seq) # batch_size x seq_len x head_size
        queries = self.query_extractor(vec_seq)
        # do not split into attention heads (as we are using just 1 attn head)
        scaled_unorm_dot_prod = (queries @ keys.transpose(1,2))/math.sqrt(self.head_size)
        # conver the binary mask to a mask which has zero at points that are not pad tokens a large negative number at tokens that are pad tokens (need to be zeroed out in the softmax).
        num_tokens = base_attn_mask.sum(axis=-1)
        batch_top_ks = (percentile_fraction*num_tokens).long()
        extended_attn_mask = self.get_extended_mask(base_attn_mask)
        # calculate token scores based on the batch_size x seq_len x seq_len attention tensor.
        # mask out pad tokens from the attention score & then normalize using a softmax.
        scaled_unorm_dot_prod += extended_attn_mask
        scaled_norm_dot_prod = self.dropout(F.softmax(
            scaled_unorm_dot_prod, dim=-1))
        # compute the token-wise scores for the sequence.
        token_scores = scaled_norm_dot_prod.sum(axis=-1)
        # normalize the token-wise scores.
        token_scores /= token_scores.sum(axis=-1).unsqueeze(dim=-1)
        # find the top-k tokens according the `batch_top_ks` based.
        explanation = torch.zeros_like(token_scores)
        for i, k in enumerate(batch_top_ks):
            explanation[i][torch.topk(token_scores[i], k=k).indices] = 1
        anti_explanation = 1-explanation

        # token-wise scores for the explanation.
        exp_token_scores = explanation * token_scores
        # re-normalize token-wise scores for explanation.
        exp_token_scores /= exp_token_scores.sum(axis=-1).unsqueeze(dim=-1)

        # token-wise scores for the anti-explanation.
        anti_exp_token_scores = anti_explanation * token_scores
        # re-normalize token-wise scores for anti-explanation.
        anti_exp_token_scores /= anti_exp_token_scores.sum(axis=-1).unsqueeze(dim=-1)
        
        # compute the weighted sum of the vector sequence based on the self attention `token_scores`.
        exp_pool = exp_token_scores.unsqueeze(dim=-1) * vec_seq
        all_seq_pool = token_scores.unsqueeze(dim=-1) * vec_seq 
        anti_exp_pool = anti_exp_token_scores.unsqueeze(dim=-1) * vec_seq
        exp_pool = exp_pool.sum(axis=1)
        all_seq_pool = all_seq_pool.sum(axis=1) 
        anti_exp_pool = anti_exp_pool.sum(axis=1)

        return token_scores, exp_pool, all_seq_pool, anti_exp_pool

