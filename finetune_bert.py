#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from model.datautils import SNLIDataset
from torch.utils.data import DataLoader
from model.bert import FaithfulBertClassifier
from transformers import BertModel, RobertaModel, DebertaModel

# seed torch, random and numpy.
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

LIST_OF_DATASETS = ["SNLI"]
LIST_OF_BERT_MODELS = ["bert", "roberta", "deberta"]
MODEL_CLASS = {
    "bert": BertModel,
    "roberta": RobertaModel,
    "deberta": DebertaModel,
}
MODEL_PATHS = {
    "bert": ["bert-base-uncased", "bert-base-cased", "bert-large-cased", "bert-large-uncased"],
    "roberta": ["roberta-base", "roberta-large"],
    "deberta": ["deberta-base", "deberta-large"],
}
def get_args():
    parser = argparse.ArgumentParser("Python script to regularly finetune BERT like models on simple classification tasks")
    parser.add_argument("-e", "--epochs", default=5, type=int, help="number of epochs for training")
    parser.add_argument("-bs", "--batch_size", default=32, type=int, help="batch size for training")
    parser.add_argument("-ls", "--log_steps", default=2000, type=int, help="number of steps after which validation is performed")
    parser.add_argument("-d", "--dataset", default="SNLI", type=str, help="the name of the dataset to be used for training")
    parser.add_argument("-tp", "--train_path", default="data/snli_1.0/snli_1.0_train.jsonl", type=str, help="path to training data")
    parser.add_argument("-vp", "--val_path", default="data/snli_1.0/snli_1.0_dev.jsonl", type=str, help="path to validation data")
    parser.add_argument("-tep", "--test_path", default="data/snli_1.0/snli_1.0_test.jsonl", type=str, help="path to testing data")
    parser.add_argument("-lr", "--learning_rate", default=1e-5, type=float, help="the learning rate for the AdamW optimizer")
    parser.add_argument("-bp", "--bert_path", type=str, default="bert-base-uncased", help="the `from_pretrained` path.")
    parser.add_argument("-bt", "--bert_type", type=str, default="bert", help="the type of BERT model to be used")
    parser.add_argument("-exp", "--exp_name", required=True, help="name of the experiment folder")
    parser.add_argument("-tqdm", "--use_tqdm", action="store_true", help="whether to use tqdm?")
    parser.add_argument("-D", "--device", type=str, default="cuda:0", help="device ID for training.")
    parser.add_argument("-nc", "--num_classes", type=int, default=2, help="number of target classes for the task")
    parser.add_argument("-tr", "--test_regular", action="store_true", help="test regular BERT")
    parser.add_argument("-m", "--margin", default=1, type=float, help="the margin used for triplet margin loss")
    parser.add_argument("-p", "--p", type=int, default=2, help="the p for the p-norm used for triplet margin loss")
    parser.add_argument("-rft", "--regular_finetune", action="store_true", help="do regular finetuning of the BERT model")
    parser.add_argument("-fft", "--faithful_finetune", action="store_true", help="faithfully finetune BERT model on data")
    parser.add_argument("-ckpt", "--checkpoint", type=str, default=None, help="checkpoint from which to resume training.")
    parser.add_argument("-cll", "--contrastive_loss_lambda", type=float, default=1, help="weighting factor for the contrastive loss.")
    # get arguments and validate them.
    args = parser.parse_args()
    assert args.dataset in LIST_OF_DATASETS # check if in the list of valid datasets.
    assert args.bert_type in LIST_OF_BERT_MODELS # check if in the list of valid BERT models.
    assert args.bert_path in MODEL_PATHS[args.bert_type] # check if a valid BERT model path for the BERT model type/class.

    return vars(args)

def validate(model, dataloader, loss_fn, **args) -> dict:
    """validate BERT and return stats."""
    tot, matches = 0, 0
    stats = {
        "preds": [],
        "batch_losses": [],
        "loss": 0, "acc": 0,
    }
    model.eval()
    valbar = tqdm(
        enumerate(dataloader), 
        total=len(dataloader), 
        disable=not(args["use_tqdm"]),
        desc=f"validating on {args['dataset']}"
    )
    for step, batch in valbar:
        with torch.no_grad():
            tot += len(batch[0])
            bert_args = [item.to(args["device"]) for item in batch[:-1]]
            logits = model(*bert_args)
            true_labels = batch[-1].to(args["device"])
            # compute the validation batch loss.
            batch_loss = loss_fn(logits, true_labels) 
            preds =  logits.argmax(axis=-1)
            # update accuracy.
            matches += (true_labels == preds).sum().item()
            # update the stats.
            stats["preds"] += preds.tolist()
            stats["batch_losses"].append(batch_loss.item())
            # update the validation bar.
            valbar.set_description(f"val: BL: {batch_loss.item():.3f} TL: {np.mean(np.mean(stats['batch_losses'])):.3f} acc: {(100*matches/tot):.2f}")
            # if step == 5: break # pour le DEBUG.
    stats["loss"] = np.mean(stats["batch_losses"])
    stats["acc"] = matches/tot

    return stats

def get_test_dataloader(**args):
    # tokenizer arguments for BERT.
    tok_args = {
        "padding": "max_length",
        "max_length": 100,
        "truncation": True,
        "return_tensors": "pt"
    }
    # load the SNLI train & val datasets.
    if args['dataset'] == "SNLI":
        test_dataset = SNLIDataset(
            args['test_path'], bert_path=args['bert_path'],
            use_tqdm=args.get("use_tqdm", False), **tok_args
        )    
    test_dataloader = DataLoader(
        test_dataset, shuffle=False, 
        batch_size=args["batch_size"],
    )

    return test_dataloader

def get_dataloaders(**args):
    # tokenizer arguments for BERT.
    tok_args = {
        "padding": "max_length",
        "max_length": 100,
        "truncation": True,
        "return_tensors": "pt"
    }
    # load the SNLI train & val datasets.
    if args['dataset'] == "SNLI":
        train_dataset = SNLIDataset(
            args['train_path'], bert_path=args['bert_path'],
            use_tqdm=args.get("use_tqdm", False), **tok_args
        )
        val_dataset = SNLIDataset(
            args['val_path'], bert_path=args['bert_path'],
            use_tqdm=args.get("use_tqdm", False), **tok_args
        )    
    # create the dataloaders.
    train_dataloader = DataLoader(
        train_dataset, shuffle=True,
        batch_size=args["batch_size"],
    )
    val_dataloader = DataLoader(
        val_dataset, shuffle=False, 
        batch_size=args["batch_size"],
    )

    return train_dataloader, val_dataloader

def test_regular_bert(**args):
    """method to test regularly finetuned BERT."""
    test_dataloader = get_test_dataloader(**args)
    loss_fn = nn.CrossEntropyLoss()
    # initialize the BERT model.
    print("loading model:")
    s = time.time()
    bert = MODEL_CLASS[args["bert_type"]].from_pretrained(args["bert_path"])
    faithful_bert = FaithfulBertClassifier(bert, num_classes=args['num_classes'])
    print(f"loaded model in {(time.time()-s):.2f}s")
    # move to the device.
    faithful_bert.to(args["device"])
    if args["checkpoint"]:
        state_dict = torch.load(args["checkpoint"], map_location="cpu")
        print(f"loading model checkpoint from: {args['checkpoint']}")
        faithful_bert.load_state_dict(state_dict)

    test_stats = validate(faithful_bert, 
                          test_dataloader, 
                          loss_fn, **args)
    # save the validation stats.
    os.makedirs(os.path.join("experiments", args['exp_name'], "test_stats"), exist_ok=True)
    test_stats_path = os.path.join(
        "experiments", args['exp_name'], 
        "test_stats", f"stats.json",
    )
    test_acc = test_stats['acc']
    with open(test_stats_path, "w") as f:
        json.dump(test_stats, f)

def finetune_bert(**args):
    """method to regularly finetune BERT like models."""
    train_dataloader, val_dataloader = get_dataloaders(**args)
    loss_fn = nn.CrossEntropyLoss()
    # initialize the BERT model.
    print("loading model:")
    s = time.time()
    bert = MODEL_CLASS[args["bert_type"]].from_pretrained(args["bert_path"])
    faithful_bert = FaithfulBertClassifier(bert, num_classes=args['num_classes'])
    print(f"loaded model in {(time.time()-s):.2f}s")
    # move to the device.
    faithful_bert.to(args["device"])
    if args["checkpoint"]:
        state_dict = torch.load(args["checkpoint"], map_location="cpu")
        print(f"loading model checkpoint from: {args['checkpoint']}")
        faithful_bert.load_state_dict(state_dict)
    # initialize the optimzier.
    optimizer = AdamW(
        faithful_bert.parameters(), eps=1e-12, 
        lr=args.get("learning_rate", 1e-5)
    )
    epochbar = tqdm(
        range(args["epochs"]), 
        desc="commencing training", 
        disable=not(args["use_tqdm"]),
    )
    # epoch loop.
    best_val_acc = 0
    for epoch in epochbar:
        # train loop.
        trainbar = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            disable=not(args["use_tqdm"]),
        )
        batch_losses = []
        tot, matches = 0, 0
        for step, batch in trainbar:
            faithful_bert.train()
            optimizer.zero_grad()
            tot += len(batch[0])
            bert_args = [item.to(args["device"]) for item in batch[:-1]]
            logits = faithful_bert(*bert_args)
            true_labels = batch[-1].to(args["device"])
            # compute the training batch loss.
            batch_loss = loss_fn(logits, true_labels) 
            # compute gradients and take optimizer step.
            batch_loss.backward()
            optimizer.step()
            # update accuracy.
            preds = logits.argmax(axis=-1)
            matches += (true_labels == preds).sum().item()
            # update the stats.
            batch_losses.append(batch_loss.item())
            # update the validation bar.
            trainbar.set_description(f"train: epoch: {epoch} BL: {batch_loss.item():.3f} TL: {np.mean(np.mean(batch_losses)):.3f} acc: {(100*matches/tot):.2f}")
            # if step == 5: break # pour le DEBUG.
            # do validation after every `log_step` steps.
            if (step + 1) % args["log_steps"] == 0:
                val_stats = validate(faithful_bert, 
                                     val_dataloader, 
                                     loss_fn, **args)
                # save the validation stats.
                os.makedirs(os.path.join(
                    "experiments", 
                    args['exp_name'], "val_stats",
                ), exist_ok=True)
                val_stats_path = os.path.join(
                    "experiments", args['exp_name'], 
                    "val_stats", f"stats_{epoch}_{step}.json",
                )
                val_acc = val_stats['acc']
                with open(val_stats_path, "w") as f:
                    json.dump(val_stats, f)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"saving model with best val acc: {val_acc} @ epoch: {epoch}")
                    torch.save(faithful_bert.state_dict(), os.path.join(
                    "experiments", args['exp_name'], 'model.pt'))

def faithful_finetune_bert(**args):
    """method to faithfully finetune BERT like models on classification tasks."""
    train_dataloader, val_dataloader = get_dataloaders(**args)
    CE_loss_fn = nn.CrossEntropyLoss()
    CL_loss_fn = nn.TripletMarginLoss(
        margin=args["margin"], 
        p=args["p"],
    )
    # initialize the BERT model.
    print("loading model:")
    s = time.time()
    bert = MODEL_CLASS[args["bert_type"]].from_pretrained(args["bert_path"])
    faithful_bert = FaithfulBertClassifier(bert, num_classes=args['num_classes'])
    print(f"loaded model in {(time.time()-s):.2f}s")
    # move to the device.
    faithful_bert.to(args["device"])
    assert args["checkpoint"] is not None
    state_dict = torch.load(args["checkpoint"], map_location="cpu")
    print(f"loading model checkpoint from: {args['checkpoint']}")
    faithful_bert.load_state_dict(state_dict)
    # freeze the bert model.
    faithful_bert.bert.eval()
    for p in faithful_bert.bert.parameters():
        p.requires_grad = False
    # initialize the optimzier.
    optimizer = AdamW(
        faithful_bert.parameters(), eps=1e-12, 
        lr=args.get("learning_rate", 1e-5)
    )
    epochbar = tqdm(
        range(args["epochs"]), 
        desc="commencing training", 
        disable=not(args["use_tqdm"]),
    )
    # epoch loop.
    best_val_acc = 0
    for epoch in epochbar:
        # train loop.
        trainbar = tqdm(
            enumerate(train_dataloader), 
            total=len(train_dataloader), 
            disable=not(args["use_tqdm"]),
        )
        batch_CE_losses = []
        batch_CL_losses = []
        batch_losses = []
        tot, matches = 0, 0
        for step, batch in trainbar:
            faithful_bert.faithful_pool.train()
            faithful_bert.faithful_classifier.train()
            optimizer.zero_grad()
            tot += len(batch[0])
            bert_args = [item.to(args["device"]) for item in batch[:-1]]
            token_scores, exp_pool, all_seq_pool, anti_exp_pool, logits = faithful_bert.faithful_forward(*bert_args)
            true_labels = batch[-1].to(args["device"])
            # compute the classifier loss over the logits (computed over explanation pool embedding).
            batch_CE_loss = CE_loss_fn(logits, true_labels) 
            # compute the contrasitve loss over the various pooled embeddings.
            # anchor: pooled over explanations.
            # positive: pooled over the whole input.
            # negative: pooled over the anti-explanation.
            batch_CL_loss = CL_loss_fn(exp_pool, all_seq_pool, anti_exp_pool)
            batch_loss = batch_CE_loss + args["contrastive_loss_lambda"]*batch_CL_loss
            # compute gradients and take optimizer step.
            batch_loss.backward()
            optimizer.step()
            # update accuracy.
            preds = logits.argmax(axis=-1)
            matches += (true_labels == preds).sum().item()
            # update the stats.
            batch_losses.append(batch_loss.item())
            batch_CE_losses.append(batch_CE_loss.item())
            batch_CL_losses.append(batch_CL_loss.item())
            # update the validation bar.
            trainbar.set_description(f"train: epoch: {epoch} BCE: {batch_CE_loss.item():.3f} TCE: {np.mean(batch_CE_losses):.3f} BCL: {batch_CL_loss.item():.3f} TCL: {np.mean(batch_CL_losses):.3f} BL: {batch_loss.item():.3f} TL: {np.mean(np.mean(batch_losses)):.3f} acc: {(100*matches/tot):.2f}")
            # if step == 5: break # pour le DEBUG.
            # do validation after every `log_step` steps.
            if (step + 1) % args["log_steps"] == 0:
                val_stats = validate(faithful_bert, 
                                     val_dataloader, 
                                     CE_loss_fn, **args)
                # save the validation stats.
                os.makedirs(os.path.join(
                    "experiments", 
                    args['exp_name'], "val_stats",
                ), exist_ok=True)
                val_stats_path = os.path.join(
                    "experiments", args['exp_name'], 
                    "val_stats", f"stats_{epoch}_{step}.json",
                )
                val_acc = val_stats['acc']
                with open(val_stats_path, "w") as f:
                    json.dump(val_stats, f)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"saving model with best val acc: {val_acc} @ epoch: {epoch}")
                    torch.save(faithful_bert.state_dict(), os.path.join(
                    "experiments", args['exp_name'], 'model.pt'))

# main function.
if __name__ == "__main__":
    cmdline_args = get_args()
    exp_folder = os.path.join(
        "experiments", 
        cmdline_args["exp_name"]
    )
    os.makedirs(exp_folder, exist_ok=True)
    config_path = os.path.join(exp_folder, "config.json")
    with open(config_path, "w") as f:
        json.dump(cmdline_args, f)
    if cmdline_args["regular_finetune"]:
        print(f"doing regular finetuning for {cmdline_args.bert_type} ({cmdline_args.bert_path}) over {cmdline_args.dataset}:")
        finetune_bert(**cmdline_args)
    if cmdline_args["faithful_finetune"]:
        faithful_finetune_bert(**cmdline_args)
    if cmdline_args["test_regular"]:
        test_regular_bert(**cmdline_args)