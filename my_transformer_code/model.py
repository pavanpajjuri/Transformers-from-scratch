#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 18:47:51 2025

@author: pavanpaj
"""

import torch
import torch.nn as nn



"""
For understanding purpose lets take "Cat Loves Milk" as an input instance in the first batch of 32 samples

Let's assume:

Batch size (N) = 32
1st Sentence: "Cat Loves Milk"
Number of words (seq_len) = 24 (max_len of a sentence in the batch)
Encoder
   Input: ([<bos>,"Cat", "Loves", "Milk",<eos> , <pad>,....]) -> value_len = key_len = query_len = max_len = 24
Decoder
   Input: ([<bos>,"Katze", "liebt", "Milch", <pad>,....]) -> value_len = key_len = query_len = max_len = 23 NO <eos> ONLY <bos>
   Target: (["Katze", "liebt", "Milch", <eos>, <pad>, ...]) -> value_len = key_len = query_len = max_len = 23 NO <bos> ONLY <eos>
   i.e. a shift of token happens for decoder to predict the next word which is the target

Embedding size (embed_size) = 512
Number of heads (heads) = 8
Head dimension (head_dim) = embed_size / heads = 512 / 8 = 64
"""

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size//heads

        assert(self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        # Each of these is a learnable weight matrix
        self.values = nn.Linear(embed_size, embed_size) # W_V (512*512 Matrix)
        self.keys = nn.Linear(embed_size, embed_size) # W_K (512*512 Matrix)
        self.queries = nn.Linear(embed_size, embed_size) # W_Q (512*512 Matrix)
        self.fc_out = nn.Linear(embed_size, embed_size) # W_O (512*512 Matrix)

    def forward(self, values, keys, query, mask):
        N = query.shape[0] # Number of Training Examples -> N = 32 in our case
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]  # 24 (23 when self attention called in Decoder) each sample length in the batch
        # Difference in Query shape and key,value shape by 1 is observed when the attention is called in Decoder AFTER masked multi head self attention i.e. at cross attention

        values = self.values(values)   # (N, value_len, embed_size) # (32, 24, 512) -> (32, 24, 512)
        keys = self.keys(keys)  # (N, key_len, embed_size)  # (32, 24, 512) -> (32, 24, 512)
        queries = self.queries(query)   # (N, query_len, embed_size)  # (32, 24 or (23 @cross atention), 512) -> (32, 24 or (23), 512)


        # Reshape into multiple heads
        values = values.reshape(N, value_len, self.heads, self.head_dim) # (32, 24, 8 , 64) nvhd
        keys = keys.reshape(N, key_len, self.heads, self.head_dim) # (32, 24, 8 , 64) nkhd
        queries = queries.reshape(N, query_len, self.heads, self.head_dim) # (32, 24, 8 , 64) nqhd


        # Einsum does matrix multiplication. for query*keys for each training example
        energy = torch.einsum("nqhd,nkhd -> nhqk", [queries, keys]) # (32, 8, 24, 24)
        # Dot product each query with each key.
        #   - We get a 24*24 (like a covariance matrix) in encoder self attention.
        #   - We get a 23*23 (like a covariance matrix) in decoder self attention.
        #   - We get a 23*24 (like a covariance matrix) in decoder cross attention.

        # This for each head. (8,24,24). On all samples in batch. (32,8,24,24)

        # Mask padded indices so their weights become close to 0   # (32, 8, 24, 24)
        if mask is not None:

            energy = energy.masked_fill(mask == 0, float("-1e20"))


        # Normalize energy values similarly to so that they sum to 1. Also divide by scaling factor for better stability
        attention = torch.softmax(energy/ (self.head_dim ** (1/2)), dim=3) # (N, heads, query_len, key_len)
        # dim = 3 indicating we operate along row across columns i.e.on query for each key

        out = torch.einsum("nhqk,nvhd->nqhd",[attention, values]) # multiplies attention scores with values. #(32, 8, 24, 24) @ (32, 24, 8, 64) → (32, 24, 8, 64)
        out = self.fc_out(out.reshape(N, query_len, self.heads * self.head_dim )) # (32, 24 (23 in decoder), 512) # Flatten the last dimensions and send it to fc_out -> # (N, query_len, embed_size)

        return out



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):

        """attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(query + attention)) # Add skip connection, run through normalization and finally dropout
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(x + forward))"""


        # Below is the Pre-LN implementation
        attention = self.attention(self.norm1(value), self.norm1(key), self.norm1(query), mask) # Apply self-attention
        x = query + self.dropout(attention)  # Skip connection with original query and dropout
        forward = self.norm2(x)  # Normalize before feedforward
        out = x + self.dropout(self.feed_forward(forward))

        return out # (32,24 (23 @ Decoder),512)


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,  # Total vocabulary size (number of unique words)
        embed_size,      # Dimension of each word embedding # 512
        num_layers,      # Number of Transformer blocks # 6
        heads,           # Number of self-attention heads # 8
        device,          # GPU/CPU device
        forward_expansion,  # Expansion factor for the feed-forward layer
        dropout,         # Dropout rate
        max_length,      # Maximum sentence length
    ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size) # Converts tokenized words into dense embeddings. i.e. each word willl be represented as embedding 1D vector -> 2D vector
        self.position_embedding = nn.Embedding(max_length, embed_size) # Adds position information (since Transformers have no recurrence).

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout = dropout,
                    forward_expansion=forward_expansion
                    )
                for _ in range(num_layers)
                ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape #(32,24) # Here x is tokenized representations of Our data # Ex: ["Cat","Loves","Milk"] -> [2(bos),7,8,9,3(eos),1(pad),1,...]. Assume N = 32 (batch), seq_length = 24 (words)
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device) #Adding positions with numbers for batch size (32,24)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions)) # (32,24,512) + (32,24,512) Making the positions learnable by creating embeddings to them
        # word embedding ("2") returns a 512 dimensional vector
        for layer in self.layers:
            out = layer(out, out, out, mask) # value, key, query (32,24,512), mask (32,1,1,24) which are the input for forward in Transformer Encoder self attention Block
            # So masking is done only on the columns i.e. last index. Specifically on the padded indexs

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads) # Masked Self-Attention
        self.transformer_block = TransformerBlock(embed_size, heads, dropout = dropout, forward_expansion = forward_expansion) # Cross-Attention + Feedforward
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, x, src_mask, trg_mask):
        #print("Here at Decoder Block sending trg_mask")
        attention = self.attention(self.norm(x), self.norm(x), self.norm(x), trg_mask) # Masked Attention -> (Lower Triangular Mask + padded Tokens Mask) = trg_mask
        query = x + self.dropout(attention)
        out = self.transformer_block(value, key, query, src_mask) # Just padding Tokens Mask like Encoder Mask

        """# Post LN Implementation
        attention = self.attention(x, x, x, trg_mask) #Uses self-attention where query = key = value = x (decoder’s past words).            # Uses trg_mask → Ensures that each word only attends to past words (not future words).
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)"""
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,  # Target vocabulary size
            embed_size,      # Dimension of each word embedding # 8
            num_layers,      # Number of Transformer blocks
            heads,           # Number of self-attention heads
            device,          # GPU/CPU device
            forward_expansion,  # Expansion factor for the feed-forward layer
            dropout,         # Dropout rate
            max_length,      # Maximum sentence length
        ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size) # Converts tokenized words into dense embeddings.
        self.position_embedding = nn.Embedding(max_length, embed_size) # Adds position information (since Transformers have no recurrence).

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    heads,
                    dropout = dropout,
                    forward_expansion=forward_expansion
                    )
                for _ in range(num_layers)
                ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, trg_vocab_size) # Projects the decoder output into target vocabulary scores


    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape  # Here x is tokenized representations of Our data # Ex: ['Katze', 'liebt', 'Milch'] -> [2(bos),7,15,18,1(pad),1,...]. Assume N = 32 (batch), seq_length = 23 (words)
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)  #(32,23)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions)) # (32,23,512) + (32,23,512)
        # word embedding ("2") returns a 512 dimensional vector
        for layer in self.layers:
            x = layer(enc_out, enc_out, x, src_mask, trg_mask) # value(32, 23 (24 @ Cross Attention), 512), key(32, 23 (24 @ Cross Attention), 512), query (32, 23, 512), mask (32, 1, 23, 23) which are the input for forward in Transformer Block

        out = self.fc_out(x) # (32, 23, 512) -> (32,23,trg_vocab_size) logit vectors, where each logit corresponds to the probability of a word in the target vocabulary.
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0.0,
        device="cpu",
        max_length=100,
    ):

        super().__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # (N = src.shape[0], 1, 1, src.shape[1])

        #tensor([[[[ True,  True,  True,  ..., True,  True,  True,  True]]],   The Cat was trying to.....in the house <eos>
        #                                 .
        #                                 .
        #                                 .
        #[[[ True,  True, True, False, False, False, False]]]])         Cats Love Milk <eos> <pad>......<pad> <pad>

        return src_mask.to(self.device)


    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        # 1. Causal (Look-Ahead) Mask (lower triangular, prevents attending to future tokens)
        causal_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool() # Shape: (trg_len, trg_len) e.g., (23, 23)

        # 2. Target Padding Mask (prevents attending to <pad> tokens in the target sequence itself) This is similar to src_mask, but for the target sequence.
        padding_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2).bool()# Shape: (N, 1, 1, trg_len) e.g., (32, 1, 1, 23)

        # 3. Combine them using logical AND (position is True only if BOTH are True)
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0) & padding_mask # (32, 1, 23, 23)
        return combined_mask.to(self.device)


    def forward(self, src, trg):
        src_mask = self.make_src_mask(src) #(32,1,1,24) Booleans # To not consider any padding values
        trg_mask = self.make_trg_mask(trg) # (32,1,23,23)  Ones and Zeros # Lower Triangualr matrix to not allow peeking and not attending to padded tokens
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == "__main__":
    
    print(torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")    
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)

    out = model(x, trg[:, :-1])
    print(out.shape)  # (2,7,10)
    
