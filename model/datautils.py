#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from typing import *
from transformers import BertTokenizer
from torch.utils.dataset import Dataset, DataLoader

# read a JSONL file.
def read_jsonl(path: str) -> List[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))

    return data

# the dataset class for SNLI.
class SNLIDataset(Dataset):
    def __init__(self, path, bert_path: str):
        self.data = read_jsonl(path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        rec = self.data[i]
        return 