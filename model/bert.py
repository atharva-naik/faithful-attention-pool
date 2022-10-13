#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import random
import numpy as np
import transformers
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from model.faithful_attn_pooling import FaithfulAttentionPooling

# set logging level of transformers.
transformers.logging.set_verbosity_error()

# A BertModel with a faithful classification head.
# bert + faithful attention pooling + linear classifier to be finetuned for any task.
class FaithfulBertClassifier(nn.Module):
    def __init__(self, bert: AutoModel, num_classes: int=2, bert_emb_size: int=768,
                faithful_head_size: int=50, faithful_dropout_rate: float=0.2):
        super(FaithfulBertClassifier, self).__init__()
        self.bert = bert
        self.bert_emb_size = bert_emb_size
        self.faithful_pool = FaithfulAttentionPooling(
            hidden_size=bert_emb_size,
            head_size=faithful_head_size,
            dropout_rate=faithful_dropout_rate,
        )
        self.faithful_classifier = nn.Linear(bert_emb_size, num_classes)
        self.classifier = nn.Linear(bert_emb_size, num_classes)

    def faithful_forward(self, *bert_args):
        """use BERT without any updates to it (gradient-wise), use the faithful pooling layer to do the pooling and finally learn a new classifier for the faithfully pooled output"""
        with torch.no_grad():
            hidden_states = self.bert(*bert_args).last_hidden_state
        base_attn_mask = bert_args[1] # the base attention mask for BERT.
        token_scores, exp_pool, all_seq_pool, anti_exp_pool = self.faithful_pool(
            hidden_states, base_attn_mask
        )
        # feed the explanation masked pooled output through the classifier to get the output.
        logits = self.faithful_classifier(exp_pool)

        return token_scores, exp_pool, all_seq_pool, anti_exp_pool, logits

    def forward(self, *bert_args):
        """use BERT in a regular fine-tuning setting and use gradients to update the BERT model too."""
        pooler_output = self.bert(*bert_args).pooler_output
        logits = self.classifier(pooler_output)

        return logits