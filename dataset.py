#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 18:53:57 2025

@author: pavanpaj
"""

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader
import random



# Tokenizers
# Tokenizer converts text into a list of tokens (words)
en_tokenizer = get_tokenizer("spacy", language = "en_core_web_sm") # Small English tokenizer model
de_tokenizer = get_tokenizer("spacy", language = "de_core_news_sm") # Small German tokenizer model


# Function to yield tokens (for vocab building) sepearate for each vocabulary
def yield_tokens(data, tokenizer, lang="en"):
    for src_text, tgt_text in data:
        text = src_text if lang == "en" else tgt_text  # Choose English or German text
        yield tokenizer(text)

# converts into a tensor of numerical token IDs -> [2,21,45,.....,3]
def text_to_tensor(text, vocab, tokenizer):
    tokens = [vocab["<bos>"]] + [vocab[token] if token in vocab else vocab["<unk>"] for token in tokenizer(text)] + [vocab["<eos>"]]
    return torch.tensor(tokens, dtype=torch.long)

# Collate function for DataLoader
def collate_fn(batch, en_vocab, de_vocab):
    src_batch, tgt_batch = [], []
    for src_text, tgt_text in batch:
        src_batch.append(text_to_tensor(src_text, en_vocab, en_tokenizer))
        tgt_batch.append(text_to_tensor(tgt_text, de_vocab, de_tokenizer))

    max_len = max(max(len(seq) for seq in src_batch), max(len(seq) for seq in tgt_batch))

    src_batch = [torch.cat([seq, torch.full((max_len - len(seq),), en_vocab["<pad>"])]) for seq in src_batch]
    tgt_batch = [torch.cat([seq, torch.full((max_len - len(seq),), de_vocab["<pad>"])]) for seq in tgt_batch]

    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)

    return src_batch, tgt_batch



if __name__ == "__main__":
    # Example:
    sample_text = "This is an example sentence."
    print("Tokenized sample:", en_tokenizer(sample_text))
    # Expected Output: ['This', 'is', 'an', 'example', 'sentence', '.']

    # Loading the data
    full_train_data = list(Multi30k(split = 'train', language_pair = ('en', 'de'))) # Multi30K is a dataset for English-German translation.
    # Shuffle the data to ensure randomness
    random.shuffle(full_train_data)

    # Split dataset into train, validation, and test (80-10-10 split)
    train_size = int(0.8 * len(full_train_data))  # 80% for training
    val_size = int(0.1 * len(full_train_data))    # 10% for validation
    test_size = len(full_train_data) - train_size - val_size  # Remaining 10% for testing

    # Create subsets
    train_data = full_train_data[:train_size]
    val_data = full_train_data[train_size:train_size + val_size]
    test_data = full_train_data[train_size + val_size:]

    print(f"Train Data: {len(train_data)} samples")
    print(f"Validation Data: {len(val_data)} samples")
    print(f"Test Data: {len(test_data)} samples")


    print("Loaded first 2 samples:", train_data[:2])  # Print first two samples


    # Build vocab for the data i.e. all unique tokens(text) to Uniques IDs from the data
    en_vocab = build_vocab_from_iterator(yield_tokens(train_data, en_tokenizer, lang = "en"), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    de_vocab = build_vocab_from_iterator(yield_tokens(train_data, de_tokenizer, lang = "de"), specials=["<unk>", "<pad>", "<bos>", "<eos>"])

    # Set unknown token index (handles words not in vocab)
    en_vocab.set_default_index(en_vocab["<unk>"])
    de_vocab.set_default_index(de_vocab["<unk>"])

    print(f"Length of English vocab is {len(en_vocab)} and German vocab is {len(de_vocab)} for {len(train_data)} samples")

    # DataLoader handles **shuffling**, **batching**, and **efficient loading**.
    batch_size = 128
    train_dataloader = DataLoader(train_data, batch_size = batch_size, collate_fn = lambda batch: collate_fn(batch, en_vocab, de_vocab))
    val_dataloader = DataLoader(val_data, batch_size = batch_size, collate_fn = lambda batch: collate_fn(batch, en_vocab, de_vocab))


    for src_batch, tgt_batch in train_dataloader:
        print(f" Source batch shape : {src_batch.shape} ; Target Batch shape : {tgt_batch.shape}")
        break


    sample_german_sentence = train_data[0][1]  # Get first German sentence
    print("German sentence:", sample_german_sentence)
    print("Tokenized:", de_tokenizer(sample_german_sentence))
    tokens = de_tokenizer("Zwei junge weiße Männer sind im Freien .")
    print("Token IDs:", [de_vocab[token] for token in tokens])

