#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from model.datautils import SNLIDataset

# tokenizer arguments for BERT.
tok_args = {
    "padding": "max_length",
    "max_length": 100,
    "truncation": True,
    "return_tensors": "pt"
}

# load the SNLI train dataset.
train_dataset = SNLIDataset("data/snli_1.0/snli_1.0_train.jsonl", use_tqdm=True, **tok_args)
print("length of train dataset:", len(train_dataset))
print(train_dataset[0])