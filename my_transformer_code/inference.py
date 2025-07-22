#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 20:05:54 2025

@author: pavanpaj
"""

import torch
import json           # For handling JSON request/response
import pickle         # To load saved vocabularies
import os             # For path manipulation
import spacy          # For tokenizers
import argparse
from torchtext.data.utils import get_tokenizer # For tokenizers
from model import Transformer 
from torchtext.data.metrics import bleu_score


# Function to load the model
def load_model(en_vocab, de_vocab, embed_size, num_layers, forward_expansion, heads, dropout, device, max_length, model_path):
    print("Loading model...")
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

    
    # Load the adjusted weights (ignore strict mismatches)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
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

def model_fn(model_dir: str) -> dict:
    """
    Loads the PyTorch model from the model_dir.
    SageMaker extracts your model.tar.gz into this directory.
    This function is called ONCE when the endpoint container starts.
    Args:
        model_dir (str): The directory containing your model artifact(s).
    Returns:
        dict: A dictionary containing the loaded model and all necessary helper objects
              (vocabs, tokenizers, device) that predict_fn will need.
    """
    print(f"Loading model for endpoint from {model_dir}")
    
    # Define hyperparams (must match training)
    # Ideally, these would be loaded from a config.json saved with the model
    # For now, hardcode to match your training script:
    embed_size = 512
    num_layers = 6
    forward_expansion = 4
    heads = 8
    dropout = 0.1 
    max_length = 256 
    
    # Determine device for inference (SageMaker endpoints typically use GPU if instance type is GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Endpoint running on device: {device}")

    # 1. Load Vocabularies
    # Try both direct path and 'model/' subfolder path for robustness based on S3 observation
    en_vocab, de_vocab = None, None
    vocab_base_path = model_dir
    vocab_sub_path = os.path.join(model_dir, "model") # Path if model.tar.gz extracts to a 'model/' folder

    for base in [vocab_base_path, vocab_sub_path]:
        try:
            with open(os.path.join(base, "en_vocab.pkl"), "rb") as f:
                en_vocab = pickle.load(f)
            with open(os.path.join(base, "de_vocab.pkl"), "rb") as f:
                de_vocab = pickle.load(f)
            print(f"Vocabularies loaded from {base}.")
            break # Exit loop if successful
        except FileNotFoundError:
            print(f"Vocab files not found in {base}. Trying next path...")
            continue
        except Exception as e:
            print(f"Error loading vocabs from {base}: {e}. Trying next path...")
            continue
    
    if en_vocab is None or de_vocab is None:
        raise RuntimeError(f"Failed to load vocabularies from {model_dir}. "
                           "Ensure en_vocab.pkl and de_vocab.pkl are correctly in model.tar.gz.")

    # 2. Re-instantiate Tokenizers (SpaCy models will be downloaded if not present)
    print("Loading spaCy models from local disk (S3 input channels)...")
    try:
        local_en_spacy_model_path = "/opt/ml/input/data/spacy_en"
        local_de_spacy_model_path = "/opt/ml/input/data/spacy_de"
        en_tokenizer = get_tokenizer(lambda text: spacy.load(local_en_spacy_model_path)(text))
        de_tokenizer = get_tokenizer(lambda text: spacy.load(local_de_spacy_model_path)(text))
        print("SpaCy tokenizers initialized successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load spaCy models from local paths: {e}. "
                           "Ensure spaCy models are correctly mounted via S3 input channels (spacy_en, spacy_de).")


    en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")
    de_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")


    # 3. Load the PyTorch model (.pth file)
    # Try both direct path and 'model/' subfolder path for robustness
    model_pth_file = None
    model_base_path = model_dir
    model_sub_path = os.path.join(model_dir, "model")

    for base in [model_base_path, model_sub_path]:
        model_candidate_path = os.path.join(base, "best_transformer_model.pth")
        if os.path.exists(model_candidate_path):
            model_pth_file = model_candidate_path
            print(f"Model .pth file found at {model_pth_file}.")
            break
        else:
            print(f"Model .pth not found in {base}. Trying next path...")
    
    if model_pth_file is None:
        raise RuntimeError(f"Failed to find best_transformer_model.pth in {model_dir}.")

    # Use your custom load_model function (which handles vocab resizing)
    model = load_model(en_vocab, de_vocab, embed_size, num_layers, forward_expansion, heads, dropout, device, max_length, model_pth_file)
    
    model.eval() # Set model to evaluation mode
    print("Model loaded into memory for inference.")
    
    return {
        'model': model,
        'en_vocab': en_vocab,
        'de_vocab': de_vocab,
        'en_tokenizer': en_tokenizer,
        'de_tokenizer': de_tokenizer,
        'device': device
    }

def compute_bleu(model, test_data, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer, device):
    model.eval()
    references = []
    hypotheses = []

    for src_sentence, trg_sentence in test_data:
        reference = [trg_tokenizer(trg_sentence)]  # Tokenize the ground-truth sentence
        hypothesis = translate_sentence(model, src_sentence, src_vocab, trg_vocab, src_tokenizer, device)

        references.append(reference)
        hypotheses.append(hypothesis)

    score = bleu_score(hypotheses, references)
    return score
    

def input_fn(request_body: str, request_content_type: str):
    """
    Deserializes the incoming HTTP request data.
    Args:
        request_body (str): The request body.
        request_content_type (str): The content type of the request.
    Returns:
        str: The extracted English sentence.
    """
    print(f"Received request with content type: {request_content_type}")
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        if 'text' not in data:
            raise ValueError("Request body must contain 'text' field for translation.")
        return data['text']
    elif request_content_type == 'text/plain':
        return request_body
    else:
        raise ValueError(f"Content type {request_content_type} not supported for input.")


def predict_fn(input_object: str, model_and_helpers: dict):
    """
    Performs inference on the deserialized input data.
    Args:
        input_object (str): The English sentence string (from input_fn).
        model_and_helpers (dict): The object returned by model_fn (model and other helpers).
    Returns:
        str: The translated German sentence.
    """
    print(f"Translating sentence: '{input_object}'")
    model = model_and_helpers['model']
    en_vocab = model_and_helpers['en_vocab']
    de_vocab = model_and_helpers['de_vocab']
    en_tokenizer = model_and_helpers['en_tokenizer']
    device = model_and_helpers['device']

    # Use your existing translate_sentence function
    translated_tokens = translate_sentence(model, input_object, en_vocab, de_vocab, en_tokenizer, device)
    translated_sentence = ' '.join(translated_tokens)
    
    return translated_sentence


def output_fn(prediction: str, accept_content_type: str):
    """
    Serializes the prediction object back into a response format.
    Args:
        prediction (str): The translated sentence from predict_fn.
        accept_content_type (str): The content type requested by the client.
    Returns:
        Tuple[str, str]: The serialized prediction and content type.
    """
    print(f"Serializing response with accept content type: {accept_content_type}")
    if accept_content_type == "application/json":
        return json.dumps({"translation": prediction}), accept_content_type
    elif accept_content_type == "text/plain":
        return prediction, accept_content_type
    else:
        print(f"Unsupported Accept header, defaulting to application/json: {accept_content_type}")
        return json.dumps({"translation": prediction}), "application/json"


# # --- Main execution block for SageMaker single translation job ---
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()

#     # SageMaker provides model artifacts in SM_MODEL_DIR
#     parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
#     # Input sentence to translate will be passed as a hyperparameter
#     parser.add_argument('--sentence-to-translate', type=str, nargs='+', default="A man is playing a guitar.")
#     # Add model hyperparameters (must match how the model was trained)
#     parser.add_argument('--embed-size', type=int, default=512)
#     parser.add_argument('--num-layers', type=int, default=6)
#     parser.add_argument('--forward-expansion', type=int, default=4)
#     parser.add_argument('--heads', type=int, default=8)
#     parser.add_argument('--dropout', type=float, default=0.1) # Dropout used during training (not active in eval)
#     parser.add_argument('--max-length', type=int, default=256) # Max sequence length used in model init

#     args = parser.parse_args()

#     # --- Device Configuration ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

#     # --- Load Data and Build Vocabularies (Must match training setup) ---
#     # SpaCy model setup (as discussed for run.py)
#     print("Setting up spaCy models...")
#     try:
#         spacy.load("en_core_web_sm")
#     except OSError:
#         print("Downloading 'en_core_web_sm' spaCy model...")
#         spacy.cli.download("en_core_web_sm")
#     try:
#         spacy.load("de_core_news_sm")
#     except OSError:
#         print("Downloading 'de_core_news_sm' spaCy model...")
#         spacy.cli.download("de_core_news_sm")
#     print("SpaCy models ready.")

#     # Load vocabs (they were saved by run.py into args.model_dir during training)
#     print("Loading vocabularies...")
#     try:
#         with open(os.path.join(args.model_dir, "en_vocab.pkl"), "rb") as f:
#             en_vocab = pickle.load(f)
#         with open(os.path.join(args.model_dir, "de_vocab.pkl"), "rb") as f:
#             de_vocab = pickle.load(f)
#         print("Vocabularies loaded successfully.")
#     except Exception as e:
#         raise RuntimeError(f"Could not load vocab files from {args.model_dir}: {e}. Ensure they were saved during training.")


#     # --- Load the Trained Model ---
#     model = load_model(
#         en_vocab, de_vocab, args.embed_size, args.num_layers, args.forward_expansion,
#         args.heads, args.dropout, device, args.max_length,
#         os.path.join(args.model_dir, "best_transformer_model.pth") # Path to .pth file within SM_MODEL_DIR
#     )
#     model.eval() # Ensure model is in eval mode for inference

#     # --- Perform Single Translation ---
#     # sentence_to_translate = args.sentence_to_translate
#     sentence_to_translate = " ".join(args.sentence_to_translate) # Join the list of words back into a sentence

#     print(f"\n--- Translating Single Sentence ---")
#     print(f"Original English: '{sentence_to_translate}'")
#     translated_output = translate_sentence(model, sentence_to_translate, en_vocab, de_vocab, en_tokenizer, device)
#     print(f"Translated German: {' '.join(translated_output)}")
#     print("Single translation job complete.")




