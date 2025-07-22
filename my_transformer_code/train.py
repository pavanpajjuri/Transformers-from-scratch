#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 19:05:21 2025

@author: pavanpaj
"""


import torch
import torch.optim as optim
from model import Transformer
from dataset import collate_fn, en_tokenizer, de_tokenizer
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator


def train(model, dataloader, optimizer, scheduler, loss_fn, device, clip):
    model.train() # Setting the model to train mode
    epoch_loss = 0
    for batch_idx, (src,tgt) in enumerate(dataloader):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()

        # Target input (remove last token) and target output (remove first token)
        tgt_input = tgt[:,:-1] # Input to decoder (without eos)
        tgt_output = tgt[:,1:] # Target output (without bos)

        # Forward pass
        output = model(src, tgt_input)


        # Reshape output to match loss function expectations
        output = output.reshape(-1, output.shape[-1])  # [batch*seq_len, vocab_size]
        tgt_output = tgt_output.reshape(-1) # [batch*seq_len]

        # Compute loss
        loss = loss_fn(output, tgt_output)
        loss.backward()

        # Gradient Clipping to ptevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)

        # Update weights
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        # # Print loss every 100 batches
        # if batch_idx % 100 == 0:
        #     print(f"Batch [{batch_idx}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

    return epoch_loss / len(dataloader)



def evaluate(model, dataloader, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    epoch_loss = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)


            tgt_input = tgt[:, :-1]  # Input to decoder (without <eos>)
            tgt_output = tgt[:, 1:]  # Target output (without <bos>)


            output = model(src, tgt_input)

            # Reshape output for loss computation
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            loss = loss_fn(output, tgt_output)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


class InverseSqrtLR(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, d_model, warmup_steps):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step_num):
        step_num = max(step_num, 1)  # Ensure step_num is at least 1
        return (self.d_model ** -0.5) * min(step_num ** -0.5, step_num * (self.warmup_steps ** -1.5))




if __name__ == "__main__":
    
    
    print("Loading dataset and Building Vocabulary...")
    train_data = list(Multi30k(split='train', language_pair=('en', 'de')))[:1000]  # Reduced for efficiency
    val_data = list(Multi30k(split='train', language_pair=('en', 'de')))[5200:5500]  # Using part of train for validation
    test_data = list(Multi30k(split='train', language_pair=('en', 'de')))[6000:6300]  # Using part of train for test

    # Build Vocabulary
    en_vocab = build_vocab_from_iterator((en_tokenizer(src) for src, _ in train_data), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    de_vocab = build_vocab_from_iterator((de_tokenizer(trg) for _, trg in train_data), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    en_vocab.set_default_index(en_vocab["<unk>"])
    de_vocab.set_default_index(de_vocab["<unk>"])
    print("Done")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")    
    print(f"Device : {device}")
    
    
    batch_size = 128
    lr= 0.01
    clip = 5
    
    train_dataloader = DataLoader(train_data, batch_size = batch_size, collate_fn = lambda batch: collate_fn(batch, en_vocab, de_vocab))
    val_dataloader = DataLoader(val_data, batch_size = batch_size, collate_fn = lambda batch: collate_fn(batch, en_vocab, de_vocab))
    
    src_pad_idx = en_vocab["<pad>"]
    trg_pad_idx = de_vocab["<pad>"]
    src_vocab_size = len(en_vocab)
    trg_vocab_size = len(de_vocab)

    # Define Transformer model
    model = Transformer(
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        device=device,
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    """
    #Adam optimizer
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    #optimizer = optim.Adam(model.parameters(), lr = 3e-4, betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.98), eps=5e-9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    """
    

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,            # Initial learning rate
        betas=(0.9, 0.98),   # β1 = 0.9, β2 = 0.98
        eps=1e-9             # Small epsilon value for numerical stability
    )


    # Set up the scheduler with warm-up steps
    warmup_steps = 4000
    scheduler = InverseSqrtLR(optimizer, d_model=512, warmup_steps=warmup_steps)
    
    print("Training for single epoch...")
    
    train_loss = train(model, train_dataloader, optimizer, scheduler, loss_fn, device, clip)
    valid_loss = evaluate(model, val_dataloader, loss_fn, device)

    print("Training Done...")
    
    
    

    

    
