#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import torch
from typing import *
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

# read a JSONL file.
def read_jsonl(path: str, use_tqdm: bool=False) -> List[dict]:
    data = []
    with open(path) as f:
        for line in tqdm(f, disable=not(use_tqdm)):
            line = line.strip()
            data.append(json.loads(line))

    return data

# the dataset class for SNLI.
class SNLIDataset(Dataset):
    def __init__(self, path, bert_path: str="bert-base-uncased",
                 use_tqdm=False, **tok_args):
        super(SNLIDataset, self).__init__()
        self.tok_args = tok_args
        self.class_map = {
            "neutral": 0,
            "entailment": 1,
            "contradiction": 2,
        }
        self.data = []
        data = read_jsonl(path, use_tqdm=use_tqdm)
        # filter out instances with "-":
        for rec in data:
            if rec["gold_label"] != "-":
                self.data.append(rec)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        rec = self.data[i]
        premise = rec["sentence1"]
        hypothesis = rec["sentence2"]
        tok_dict = self.tokenizer(premise, hypothesis, **self.tok_args)
        input_ids = torch.as_tensor(tok_dict["input_ids"][0])
        attention_mask = torch.as_tensor(tok_dict["attention_mask"][0])
        gold_label = torch.tensor(self.class_map[rec["gold_label"]])
        if "token_type_ids" in tok_dict:
            token_type_ids = torch.as_tensor(tok_dict["token_type_ids"][0])
            return [input_ids, attention_mask, token_type_ids, gold_label]
        else:
            return [input_ids, attention_mask, gold_label]