#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 19:28:46 2025

@author: pavanpaj
"""
import torch
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator

from train import train, evaluate, InverseSqrtLR
from dataset import collate_fn, en_tokenizer, de_tokenizer
from model import Transformer
from torch.utils.data import DataLoader
from inference import translate_sentence, load_model, compute_bleu

import random
import time
import matplotlib.pyplot as plt


random.seed(69)

if __name__ == "__main__":
    
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


    #print("Loaded first 2 samples:", train_data[:2])  # Print first two samples
    
    # Build Vocabulary
    en_vocab = build_vocab_from_iterator((en_tokenizer(src) for src, _ in train_data), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    de_vocab = build_vocab_from_iterator((de_tokenizer(trg) for _, trg in train_data), specials=["<unk>", "<pad>", "<bos>", "<eos>"])
    en_vocab.set_default_index(en_vocab["<unk>"])
    de_vocab.set_default_index(de_vocab["<unk>"])
    

    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")    
    print(f"Device : {device}")
    
    
    num_epochs = 2
    clip = 5 # Gradient Clipping to prevent exploding gradients
    embed_size = 512
    num_layers=6
    forward_expansion=4
    heads=8
    dropout=0.1
    device=device
    max_length=256
    batch_size = 128
    lr = 0.001
    

    train_dataloader = DataLoader(train_data, batch_size = batch_size, collate_fn = lambda batch: collate_fn(batch, en_vocab, de_vocab))
    val_dataloader = DataLoader(val_data, batch_size = batch_size, collate_fn = lambda batch: collate_fn(batch, en_vocab, de_vocab))
    print(f"Length of English vocab is {len(en_vocab)} and German vocab is {len(de_vocab)} for {len(train_data)} samples")
    
    
    print("Training...")
    
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
        embed_size=embed_size,
        num_layers=num_layers,
        forward_expansion=forward_expansion,
        heads=heads,
        dropout=dropout,
        device=device,
        max_length=max_length,
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=src_pad_idx)
    """
    #Adam optimizer
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    #optimizer = optim.Adam(model.parameters(), lr = 3e-4, betas=(0.9, 0.98), eps=1e-9)
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.98), eps=5e-9, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    """
    

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,            # Initial learning rate
        betas=(0.9, 0.98),   # β1 = 0.9, β2 = 0.98
        eps=1e-9             # Small epsilon value for numerical stability
    )


    # Set up the scheduler with warm-up steps
    warmup_steps = 4000
    scheduler = InverseSqrtLR(optimizer, d_model=512, warmup_steps=warmup_steps)
    
    model_path = "best_transformer_model.pth"


    # Run training & evaluation
    best_valid_loss = float("inf")
    # Store loss values for visualization
    train_losses = []
    valid_losses = []


    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_dataloader, optimizer, scheduler, loss_fn, device, clip)
        valid_loss = evaluate(model, val_dataloader, loss_fn, device)
        # scheduler.step(valid_loss) # Step the scheduler based on validation loss # took care in train function

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(int(end_time - start_time), 60)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        torch.save(model.state_dict(), model_path)
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.3f} | Validation Loss: {valid_loss:.3f} Time: {epoch_mins}m {epoch_secs}s")

        # # Save the model if validation loss improves
        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), "best_transformer_model.pth")
        #     print("Model saved!")

    # Plot Training & Validation Loss After Training
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss Over Epochs')
    plt.legend()
    plt.show()
    print("Training Done...")
    
    print("Testing...")
    model_path = "/Users/pavanpaj/UB Courses/MyLearnings/Transformers/Project/best_transformer_model_300.pth"
    model = load_model(en_vocab, de_vocab, embed_size, num_layers, forward_expansion, heads, dropout, device, max_length, model_path)
    sentence =  "A man is playing guitar."
    print(f"Translating  Sentence '{sentence}' ->", translate_sentence(model, sentence, en_vocab, de_vocab, en_tokenizer, device))
    
    test_loader = DataLoader(test_data[:5], batch_size = batch_size, collate_fn = lambda batch: collate_fn(batch, en_vocab, de_vocab))
    # Compute BLEU score
    bleu_score, references, translations = compute_bleu(model, test_loader, en_vocab, de_vocab, en_tokenizer, device = device)

    # Print Results
    print(f"BLEU Score: {bleu_score:.2f}")
    print("\nSample Translations:")
    for i in range(min(5, len(test_data))):  # Print first 5 examples
        print(f"Original : {references[i][0]}")
        print(f"Generated: {translations[i]}\n")
    
    
    
    
    
    
    
    