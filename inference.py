#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:05:54 2025

@author: pavanpaj
"""

import torch
import os
import sacrebleu  # For BLEU Score
from model import Transformer


# Function to load the model
def load_model(en_vocab, de_vocab, embed_size, num_layers, forward_expansion, heads, dropout, device, max_length, model_path):
    print("Loading model...")
    
    
    checkpoint = torch.load(model_path, map_location=device)

    model = Transformer(len(en_vocab), len(de_vocab), en_vocab["<pad>"], de_vocab["<pad>"], 
                        embed_size=embed_size,
                        num_layers=num_layers,
                        forward_expansion=forward_expansion,
                        heads=heads,
                        dropout=dropout,
                        device=device,
                        max_length=max_length).to(device)
    
        # Get the model's current embedding weights
    model_state_dict = model.state_dict()
    
    # Resize encoder embedding if vocab size changed
    if checkpoint["encoder.word_embedding.weight"].shape != model_state_dict["encoder.word_embedding.weight"].shape:
        print("Resizing Encoder Embedding Layer...")
        old_embedding = checkpoint["encoder.word_embedding.weight"]
        new_embedding = model_state_dict["encoder.word_embedding.weight"]
        min_size = min(old_embedding.shape[0], new_embedding.shape[0])
        new_embedding[:min_size, :] = old_embedding[:min_size, :]
        checkpoint["encoder.word_embedding.weight"] = new_embedding
    
    # Resize decoder embedding if vocab size changed
    if checkpoint["decoder.word_embedding.weight"].shape != model_state_dict["decoder.word_embedding.weight"].shape:
        print("Resizing Decoder Embedding Layer...")
        old_embedding = checkpoint["decoder.word_embedding.weight"]
        new_embedding = model_state_dict["decoder.word_embedding.weight"]
        min_size = min(old_embedding.shape[0], new_embedding.shape[0])
        new_embedding[:min_size, :] = old_embedding[:min_size, :]
        checkpoint["decoder.word_embedding.weight"] = new_embedding
    
    # Resize output layer if vocab size changed
    if checkpoint["decoder.fc_out.weight"].shape != model_state_dict["decoder.fc_out.weight"].shape:
        print("Resizing Decoder Output Layer...")
        old_output = checkpoint["decoder.fc_out.weight"]
        new_output = model_state_dict["decoder.fc_out.weight"]
        min_size = min(old_output.shape[0], new_output.shape[0])
        new_output[:min_size, :] = old_output[:min_size, :]
        checkpoint["decoder.fc_out.weight"] = new_output
    
    # Resize output bias if vocab size changed
    if checkpoint["decoder.fc_out.bias"].shape != model_state_dict["decoder.fc_out.bias"].shape:
        print("Resizing Decoder Output Bias...")
        old_bias = checkpoint["decoder.fc_out.bias"]
        new_bias = model_state_dict["decoder.fc_out.bias"]
        min_size = min(old_bias.shape[0], new_bias.shape[0])
        new_bias[:min_size] = old_bias[:min_size]
        checkpoint["decoder.fc_out.bias"] = new_bias
    
    # Load the adjusted weights (ignore strict mismatches)
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    
    return model



def translate_sentence(model, sentence, src_vocab, trg_vocab, tokenizer, device, max_length=50):
    model.eval()  # Set model to evaluation mode
    
    # Tokenize and convert to tensor safely
    tokens = [src_vocab["<bos>"]] + [
        src_vocab[token] if token in src_vocab else src_vocab["<unk>"] for token in tokenizer(sentence)
    ] + [src_vocab["<eos>"]]

    # Debugging: Check if any tokens are out of vocab
    for token in tokenizer(sentence):
        if token not in src_vocab:
            print(f"WARNING: Token '{token}' not in vocab. Mapping to <unk>.")

    # Convert tokens to tensor and add batch dimension
    sentence_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  

    #print(f"\n Tokenized Sentence: {tokens}")
    #print(f" Sentence Tensor Shape: {sentence_tensor.shape}")
    #print(f" Max Index in Sentence: {max(tokens)}, Vocab Size: {len(src_vocab)}")

    with torch.no_grad():
        # Encode the source sentence
        src_mask = model.make_src_mask(sentence_tensor)
        enc_src = model.encoder(sentence_tensor, src_mask)

        # Initialize target sequence with <bos>
        trg_indexes = [trg_vocab["<bos>"]]

        #print(f" BOS Index: {trg_vocab['<bos>']}, EOS Index: {trg_vocab['<eos>']}")

        for i in range(max_length):
            trg_tensor = torch.tensor(trg_indexes, dtype=torch.long).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)

            # Pass through the decoder
            output = model.decoder(trg_tensor, enc_src, src_mask, trg_mask)

            # Get the next token (greedy decoding)
            next_word = output.argmax(2)[:, -1].item()
            trg_indexes.append(next_word)

            # Debugging: Print top predictions
            #probs = torch.softmax(output[:, -1, :], dim=-1)
            #top5 = torch.topk(probs, 5)
            #print(f"Step {i+1}: Next token: {next_word} ({trg_vocab.get_itos()[next_word]}) | Top5 Predictions: {top5.indices.tolist()}")

            # Stop if <eos> is generated
            if next_word == trg_vocab["<eos>"]:
                #print(" Stopping as <eos> is generated.")
                break

    translated_tokens = [trg_vocab.get_itos()[idx] for idx in trg_indexes]
    #print(f" Translated Tokens: {translated_tokens[1:-1]}")  # Excluding <bos> and <eos>
    return translated_tokens[1:-1]  # Remove <bos> and <eos>

def compute_bleu(model, test_loader, src_vocab, trg_vocab, en_tokenizer, device):
    model.eval()
    references = []
    translations = []

    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)

            for i in range(len(src)):  # Iterate over batch
                src_sentence = " ".join([src_vocab.get_itos()[idx] for idx in src[i].tolist()
                                         if idx not in {src_vocab["<bos>"], src_vocab["<eos>"], src_vocab["<pad>"]}])
                tgt_sentence = " ".join([trg_vocab.get_itos()[idx] for idx in tgt[i].tolist()
                                         if idx not in {trg_vocab["<bos>"], trg_vocab["<eos>"], trg_vocab["<pad>"]}])

                # Translate the sentence
                translated_tokens = translate_sentence(model, src_sentence, src_vocab, trg_vocab, en_tokenizer, device)
                translated_sentence = " ".join(translated_tokens)  # Ensure it's a string

                references.append([tgt_sentence])  # BLEU requires list of lists
                translations.append(translated_sentence)  # Ensure it's a string

    bleu = sacrebleu.corpus_bleu(translations, references).score
    return bleu, references, translations
