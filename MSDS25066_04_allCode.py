
# data_utils.py
# PURPOSE : Load, clean, tokenize and prepare Stack Overflow data
# Author  : Saad Ali (MSDS25066)
# Dataset : Stack Overflow Python Q&A

import pandas as pd
import numpy as np
import re
import os
import torch
from torch.utils.data import Dataset, DataLoader
from bs4 import BeautifulSoup
from bs4 import MarkupResemblesLocatorWarning
import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


# ============================================================
# SECTION 1: LOAD AND MERGE DATA
# ============================================================
# CONCEPT: Questions and Answers are in separate files.
# They are linked by:
#   Questions.Id  ==  Answers.ParentId
#
# Since we have no AcceptedAnswerId, we pick the
# HIGHEST SCORED answer for each question.
# This is a reasonable proxy for "best answer".
# ============================================================

def load_and_merge_data(questions_path, answers_path, num_samples=5000):
    """
    Load Questions.csv and Answers.csv, then merge them.
    
    Steps:
    1. Load both CSVs
    2. For each question, find its highest scored answer
    3. Merge into one DataFrame with columns: title, body, answer
    4. Take only num_samples rows (to save memory/compute)
    
    Args:
        questions_path : path to Questions.csv
        answers_path   : path to Answers.csv
        num_samples    : how many QA pairs to use (keep small!)
    
    Returns:
        DataFrame with columns [title, body, answer]
    """
    print(f"Loading data... (using {num_samples} samples)")

    # Load CSVs with latin1 encoding (required for this dataset)
    questions = pd.read_csv(questions_path, encoding='latin1')
    answers   = pd.read_csv(answers_path,   encoding='latin1')

    print(f"  Loaded {len(questions)} questions")
    print(f"  Loaded {len(answers)} answers")

    # ---------------------------------------------------------
    # STEP 1: For each question, get the BEST answer
    # "best" = highest Score
    # ---------------------------------------------------------
    # Sort answers by Score descending so highest is first
    answers_sorted = answers.sort_values('Score', ascending=False)

    # Keep only the FIRST (highest scored) answer per question
    # drop_duplicates keeps the first occurrence of each ParentId
    best_answers = answers_sorted.drop_duplicates(
        subset='ParentId', keep='first'
    )

    print(f"  Unique questions with answers: {len(best_answers)}")

    # ---------------------------------------------------------
    # STEP 2: MERGE questions with their best answers
    # We join on: questions.Id == best_answers.ParentId
    # ---------------------------------------------------------
    merged = pd.merge(
        questions,                    # Left table
        best_answers[['ParentId',     # Right table (only need these cols)
                      'Body']],
        left_on  = 'Id',              # Join key from questions
        right_on = 'ParentId',        # Join key from answers
        how      = 'inner'            # Only keep rows that matched
    )

    # Rename columns to be clear
    merged = merged.rename(columns={
        'Body_x' : 'question_body',   # Question body
        'Body_y' : 'answer_body',     # Answer body
        'Title'  : 'title'
    })

    # Keep only what we need
    merged = merged[['title', 'question_body', 'answer_body']]

    # Drop rows where any field is missing
    merged = merged.dropna()

    print(f"  Total QA pairs after merge: {len(merged)}")

    # ---------------------------------------------------------
    # STEP 3: Take limited samples (as per assignment)
    # Shuffle first so we get diverse samples
    # random_state=42 means same shuffle every time (reproducible)
    # ---------------------------------------------------------
    merged = merged.sample(
        n   = min(num_samples, len(merged)),
        random_state = 42
    ).reset_index(drop=True)

    print(f"  Using {len(merged)} samples for training")
    return merged


# ============================================================
# SECTION 2: TEXT CLEANING
# ============================================================
# CONCEPT: Raw Stack Overflow text contains:
# - HTML tags like <p>, <code>, <br>  → not meaningful for model
# - Special characters → add noise
# - Mixed case → "Python" and "python" treated as different words
#
# We clean all of this before feeding to the model.
# ============================================================

def clean_text(text):
    """
    Clean a single piece of text.
    
    Operations (in order):
    1. Handle non-string input
    2. Remove HTML tags using BeautifulSoup
    3. Convert to lowercase
    4. Remove special characters (keep letters, numbers, basic punctuation)
    5. Remove extra whitespace
    
    Example:
        Input  : "<p>How do I use <code>pandas</code>?</p>"
        Output : "how do i use pandas?"
    
    Args:
        text : raw string
    Returns:
        cleaned string
    """

    # Step 1: Handle non-string (NaN, float, etc.)
    if not isinstance(text, str):
        return ""

    # Step 2: Remove HTML tags
    # BeautifulSoup parses HTML, .get_text() extracts only visible text
    # "html.parser" is Python's built-in HTML parser (no extra install needed)
    text = BeautifulSoup(text, 'html.parser').get_text(separator=' ')

    # Step 3: Lowercase everything
    # "Python" and "python" mean the same → treat them identically
    text = text.lower()

    # Step 4: Remove special characters
    # re.sub = regular expression substitution
    # Pattern [^a-z0-9\s.,?!] means:
    #   ^ inside [] = "NOT"
    #   a-z = keep letters
    #   0-9 = keep numbers
    #   \s  = keep spaces
    #   .,?!= keep basic punctuation
    # Everything else gets replaced with a space
    text = re.sub(r'[^a-z0-9\s.,?!]', ' ', text)

    # Step 5: Collapse multiple spaces into one, strip edges
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================
# SECTION 3: FORMAT AS Q&A SEQUENCE
# ============================================================
# CONCEPT: The model needs to learn the PATTERN:
#
#   <START> question text [SEP] answer text <END>
#
# The special tokens act as anchors:
# - <START> tells model "a new QA pair begins"
# - [SEP]   tells model "question ended, answer starts"
# - <END>   tells model "this QA pair is complete"
#
# During inference (answer generation), we give:
#   <START> new question [SEP]
# And the model continues from there to generate the answer.
# ============================================================

def format_qa_pair(title, question_body, answer_body,
                   name="SaadAli", roll="MSDS25066"):
    """
    Format one QA pair as a single training string.
    
    Format:
    <SaadAli_MSDS25066> title . question_body [SEP] answer_body </SaadAli_MSDS25066>
    
    Args:
        title         : question title (string)
        question_body : question description (string)
        answer_body   : answer text (string)
        name          : your first name
        roll          : your roll number
    
    Returns:
        formatted string ready for tokenization
    """
    start_token = f"<{name}_{roll}>"
    end_token   = f"</{name}_{roll}>"
    sep_token   = "[SEP]"

    # Clean all three parts
    clean_title    = clean_text(title)
    clean_question = clean_text(question_body)
    clean_answer   = clean_text(answer_body)

    # Combine: START + title + question + SEP + answer + END
    formatted = (
        f"{start_token} "
        f"{clean_title} . {clean_question} "
        f"{sep_token} "
        f"{clean_answer} "
        f"{end_token}"
    )

    return formatted


# ============================================================
# SECTION 4: PYTORCH DATASET CLASS
# ============================================================
# CONCEPT: PyTorch needs data in a specific format.
# A Dataset class tells PyTorch:
#   __len__     → how many samples total?
#   __getitem__ → give me sample number i
#
# The key idea for LANGUAGE MODELING:
#   Input  = tokens[0 : seq_len-1]   (all except last)
#   Target = tokens[1 : seq_len]     (all except first)
#
# This creates a "shifted" pair:
#   If tokens = [A, B, C, D, E]
#   Input      = [A, B, C, D]    ← feed to model
#   Target     = [B, C, D, E]    ← model should predict
#
# So at position 0, model sees A → should predict B
# At position 1, model sees A,B → should predict C
# This is how the model learns to generate text!
# ============================================================

class QADataset(Dataset):
    """
    PyTorch Dataset for Stack Overflow QA pairs.
    
    Args:
        texts      : list of formatted QA strings
        tokenizer  : HuggingFace tokenizer
        max_length : max sequence length (pad/truncate to this)
    """

    def __init__(self, texts, tokenizer, max_length=256):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.data       = texts

        print(f"  Dataset created with {len(texts)} samples")
        print(f"  Max sequence length: {max_length}")

    def __len__(self):
        # PyTorch calls this to know dataset size
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return one training sample as (input, target) tensors.
        
        Both are token ID sequences of length (max_length - 1).
        Target is input shifted one position to the right.
        """
        text = self.data[idx]

        # Tokenize: convert string → list of token IDs
        # padding    = add [PAD] tokens to reach max_length
        # truncation = cut if longer than max_length
        # return_tensors='pt' = return PyTorch tensors
        encoding = self.tokenizer(
            text,
            max_length    = self.max_length,
            padding       = 'max_length',
            truncation    = True,
            return_tensors= 'pt'
        )

        # Squeeze removes the batch dimension added by return_tensors
        # Shape goes from (1, max_length) → (max_length,)
        token_ids = encoding['input_ids'].squeeze()

        # Create input/target pair by shifting
        # input  = all tokens EXCEPT the last one
        # target = all tokens EXCEPT the first one
        x = token_ids[:-1]   # Shape: (max_length - 1,)
        y = token_ids[1:]    # Shape: (max_length - 1,)

        return x, y


# ============================================================
# SECTION 5: CREATE DATA SPLITS AND LOADERS
# ============================================================
# CONCEPT: We split data into 3 sets:
#
#   Train (80%)      → model learns from this
#   Validation (10%) → monitor during training (early stopping)
#   Test (10%)       → final evaluation ONLY (never seen during training)
#
# WHY separate test set?
# If you evaluate on training data, model looks great but actually
# just memorized the answers (overfitting).
# Test set reveals TRUE performance on unseen data.
#
# Data Leakage: we shuffle BEFORE splitting to ensure
# no ordering bias. Each split has completely different samples.
# ============================================================

def create_dataloaders(texts, tokenizer,
                       max_length=256, batch_size=16):
    """
    Split texts into train/val/test and create DataLoaders.
    
    Args:
        texts      : list of all formatted QA strings
        tokenizer  : HuggingFace tokenizer
        max_length : sequence length for padding/truncation
        batch_size : samples per batch during training
    
    Returns:
        train_loader, val_loader, test_loader
    """

    print(f"\nSplitting {len(texts)} samples...")

    # Split 1: 80% train, 20% temporary
    # shuffle=True prevents ordering bias
    # random_state=42 makes split reproducible
    train_texts, temp_texts = train_test_split(
        texts,
        test_size    = 0.2,
        random_state = 42,
        shuffle      = True
    )

    # Split 2: Split the 20% temp into 10% val + 10% test
    val_texts, test_texts = train_test_split(
        temp_texts,
        test_size    = 0.5,
        random_state = 42
    )

    print(f"  Train samples      : {len(train_texts)}")
    print(f"  Validation samples : {len(val_texts)}")
    print(f"  Test samples       : {len(test_texts)}")

    # Create Dataset objects
    train_dataset = QADataset(train_texts, tokenizer, max_length)
    val_dataset   = QADataset(val_texts,   tokenizer, max_length)
    test_dataset  = QADataset(test_texts,  tokenizer, max_length)

    # Create DataLoaders
    # DataLoader handles:
    #   - Batching: groups samples into batches
    #   - Shuffling: randomize order each epoch (train only)
    #   - num_workers: parallel data loading (0 = main process)
    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,       # Shuffle training data each epoch
        num_workers = 0           # Keep 0 on Windows to avoid errors
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,      # No shuffle for validation
        num_workers = 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size,
        shuffle     = False,      # No shuffle for test
        num_workers = 0
    )

    return train_loader, val_loader, test_loader


# ============================================================
# SECTION 6: MAIN PIPELINE FUNCTION
# ============================================================
# This combines all steps above into one clean function
# that other files (train.py) will call.
# ============================================================

def prepare_data(questions_path, answers_path,
                 num_samples=5000, max_length=256, batch_size=16):
    """
    Full data preparation pipeline.
    
    Combines all steps:
    1. Load and merge CSVs
    2. Format as QA pairs
    3. Initialize tokenizer
    4. Create datasets and dataloaders
    
    Args:
        questions_path : path to Questions.csv
        answers_path   : path to Answers.csv
        num_samples    : how many samples to use
        max_length     : sequence length
        batch_size     : batch size for training
    
    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """

    # STEP 1: Load data
    df = load_and_merge_data(questions_path, answers_path, num_samples)

    # STEP 2: Format all rows as QA strings
    print("\nFormatting QA pairs...")
    texts = []
    for idx, row in df.iterrows():
        formatted = format_qa_pair(
            title         = row['title'],
            question_body = row['question_body'],
            answer_body   = row['answer_body']
        )
        texts.append(formatted)

    # Show a sample formatted text
    print("\nSample formatted text (first 300 chars):")
    print(texts[0][:300])

    # STEP 3: Initialize tokenizer
    # openai-gpt tokenizer is good for this task
    # It handles English text well and has a reasonable vocab size
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "openai-community/openai-gpt"
    )

    # GPT tokenizer has no padding token by default
    # We add a special [PAD] token manually
    # This is the correct way to handle GPT-style tokenizers
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print(f"  Vocabulary size : {len(tokenizer)}")
    print(f"  Pad token       : {tokenizer.pad_token}")
    print(f"  Pad token ID    : {tokenizer.pad_token_id}")

    # STEP 4: Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        texts, tokenizer, max_length, batch_size
    )

    print("\nData preparation complete!")
    return train_loader, val_loader, test_loader, tokenizer


# ============================================================
# QUICK TEST - run this file directly to verify everything works
# ============================================================
if __name__ == "__main__":

    print("Testing data_utils.py...")
    print("="*60)

    train_loader, val_loader, test_loader, tokenizer = prepare_data(
        questions_path = 'dataset/Questions.csv',
        answers_path   = 'dataset/Answers.csv',
        num_samples    = 100,    # Use only 100 for quick test
        max_length     = 128,
        batch_size     = 4
    )

    # Test one batch
    print("\nTesting one batch...")
    x_batch, y_batch = next(iter(train_loader))
    print(f"  Input batch shape  : {x_batch.shape}")
    print(f"  Target batch shape : {y_batch.shape}")
    print(f"  Expected shape     : (4, 127)")

    # Decode one sample to verify it looks right
    print("\nDecoding first sample to verify...")
    decoded = tokenizer.decode(x_batch[0], skip_special_tokens=False)
    print(f"  First 200 chars: {decoded[:200]}")

    print("\ndata_utils.py is working correctly!")

# model.py
# PURPOSE  : Transformer model architecture for QA generation
# Author   : Saad Ali (MSDS25066)
# Based on : "Attention is All You Need" (Vaswani et al., 2017)
#
# ARCHITECTURE OVERVIEW:
#
#   Token IDs
#       ↓
#   [Embedding Layer]        converts token IDs → dense vectors
#       ↓
#   [Positional Encoding]    adds position information to vectors
#       ↓
#   [Transformer Block] x N  core learning happens here
#   │
#   ├── [Masked Multi-Head Self-Attention]
#   ├── [Add & LayerNorm]
#   ├── [FeedForward Network]
#   └── [Add & LayerNorm]
#       ↓
#   [Final LayerNorm]
#       ↓
#   [Linear Projection]      maps vectors → vocabulary scores
#       ↓
#   Predicted next token probabilities

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# COMPONENT 1: POSITIONAL ENCODING
# ============================================================
# PROBLEM: Transformers process ALL tokens simultaneously.
# Unlike RNNs, they have no built-in sense of order.
# "cat sat on mat" would look same as "mat on sat cat"!
#
# SOLUTION: Add a unique position signal to each token's vector.
#
# MATH:
#   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
#   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# Where:
#   pos     = position in sequence (0, 1, 2, ...)
#   i       = dimension index
#   d_model = size of embedding vector
#
# WHY sin/cos?
# - They create unique patterns for each position
# - The model can learn relative positions from them
# - Values stay between -1 and 1 (no scaling issues)
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings.

    Args:
        d_model     : embedding dimension (must match model)
        max_seq_len : maximum sequence length supported
        dropout     : dropout rate for regularization
    """

    def __init__(self, d_model, max_seq_len=512, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Create empty matrix: rows=positions, cols=dimensions
        # Shape: (max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)

        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        # unsqueeze(1) adds a column dimension → shape (max_seq_len, 1)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        # Denominator term: 10000^(2i/d_model)
        # Using exp(log) form for numerical stability
        # torch.arange(0, d_model, 2) = [0, 2, 4, ..., d_model-2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # Apply sin to even-indexed dimensions (0, 2, 4, ...)
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cos to odd-indexed dimensions (1, 3, 5, ...)
        # Handle case where d_model is odd
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        # Add batch dimension: (1, max_seq_len, d_model)
        # This lets it broadcast over any batch size
        pe = pe.unsqueeze(0)

        # register_buffer: saves pe with model but NOT as learnable parameter
        # It moves to GPU automatically when model.to(device) is called
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input embeddings.

        Args:
            x : token embeddings, shape (batch, seq_len, d_model)
        Returns:
            x + positional encoding, same shape
        """
        # x.size(1) = current sequence length
        # pe[:, :seq_len, :] selects only needed positions
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# COMPONENT 2: SCALED DOT-PRODUCT ATTENTION
# ============================================================
# This is the CORE mathematical operation of transformers.
#
# INTUITION:
#   Imagine you're at a library.
#   - Query (Q)  = what you're looking for ("machine learning books")
#   - Key   (K)  = labels on each shelf ("cooking", "ML", "history")
#   - Value (V)  = actual content on each shelf
#
#   You compare your query against all keys,
#   find the most relevant shelves,
#   then retrieve content from those shelves.
#
# MATH:
#   Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V
#
#   Q @ K^T  → similarity scores between queries and keys
#   / sqrt(d_k) → scale to prevent vanishing gradients
#   softmax  → convert scores to probabilities (sum to 1)
#   @ V      → weighted sum of values
#
# CAUSAL MASK:
#   During training, we must prevent position i from
#   "seeing" future positions i+1, i+2, ...
#   Otherwise the model just copies the answer instead of learning!
#   We set future positions to -infinity before softmax
#   so they become 0 after softmax.
# ============================================================

def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q    : Query matrix,  shape (batch, heads, seq_len, d_k)
        K    : Key matrix,    shape (batch, heads, seq_len, d_k)
        V    : Value matrix,  shape (batch, heads, seq_len, d_k)
        mask : Causal mask,   shape (seq_len, seq_len) - optional
        dropout : dropout layer - optional

    Returns:
        output          : shape (batch, heads, seq_len, d_k)
        attention_weights: shape (batch, heads, seq_len, seq_len)
    """

    # d_k = dimension of each key/query vector
    d_k = Q.size(-1)

    # Step 1: Compute raw attention scores
    # Q @ K^T = how much does each query match each key?
    # Shape: (batch, heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Scale by sqrt(d_k)
    # WHY? Without scaling, dot products grow large for big d_k
    # making softmax output near 0 or 1 (vanishing gradients)
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply causal mask (prevent seeing future tokens)
    # mask=0 at positions we want to block
    # We fill those with -1e9 (very negative number)
    # After softmax, e^(-1e9) ≈ 0, so those positions get no attention
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Step 4: Softmax → convert scores to probabilities
    # dim=-1 means softmax over the last dimension (key positions)
    # Each row sums to 1: "how much attention to pay to each position"
    attention_weights = F.softmax(scores, dim=-1)

    # Step 5: Apply dropout to attention weights (regularization)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Step 6: Weighted sum of values
    # Each output position = weighted combination of all value vectors
    # Shape: (batch, heads, seq_len, d_k)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# ============================================================
# COMPONENT 3: MULTI-HEAD ATTENTION
# ============================================================
# CONCEPT: Instead of one attention, run h attention heads
# in PARALLEL, each looking at different aspects of the text.
#
# WHY MULTIPLE HEADS?
#   Head 1 might learn: subject → verb relationships
#   Head 2 might learn: pronouns → their references
#   Head 3 might learn: questions → key answer words
#   Head 4 might learn: local word context
#
# MATH:
#   MultiHead(Q,K,V) = Concat(head_1,...,head_h) @ W_o
#   where head_i = Attention(Q @ W_qi, K @ W_ki, V @ W_vi)
#
# Each head has its own learned projection matrices (W_q, W_k, W_v)
# Final W_o combines all heads back to d_model dimensions
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.

    Args:
        d_model   : total embedding dimension
        num_heads : number of parallel attention heads
        dropout   : dropout rate
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        # Sanity check: d_model must divide evenly into heads
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads  # dimension per head

        # Learned linear projections for Q, K, V, and output
        # These are the W_q, W_k, W_v, W_o matrices from the paper
        # Each maps d_model → d_model
        # (internally each head gets d_model/num_heads = d_k dimensions)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Apply multi-head attention to input.

        Args:
            x    : input tensor, shape (batch, seq_len, d_model)
            mask : causal mask,  shape (seq_len, seq_len)

        Returns:
            output : shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Step 1: Project input to Q, K, V
        # Each has shape (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 2: Split into multiple heads
        # Reshape: (batch, seq_len, d_model)
        #       → (batch, seq_len, num_heads, d_k)
        #       → (batch, num_heads, seq_len, d_k)  [transpose]
        # Now each head can process independently
        Q = Q.view(batch_size, seq_len,
                   self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len,
                   self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len,
                   self.num_heads, self.d_k).transpose(1, 2)

        # Step 3: Apply attention for ALL heads simultaneously
        # PyTorch handles the batch and head dimensions automatically
        attn_output, _ = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        # attn_output shape: (batch, num_heads, seq_len, d_k)

        # Step 4: Concatenate all heads back together
        # (batch, num_heads, seq_len, d_k)
        # → (batch, seq_len, num_heads, d_k)  [transpose]
        # → (batch, seq_len, d_model)          [reshape]
        # contiguous() required before view() after transpose
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)

        # Step 5: Final linear projection (W_o)
        # Combines information from all heads
        output = self.W_o(attn_output)

        return output


# ============================================================
# COMPONENT 4: FEED FORWARD NETWORK
# ============================================================
# After attention, each position passes through a small
# 2-layer neural network INDEPENDENTLY.
#
# This adds non-linearity and "memory" to the model.
# Think of it as: attention finds relevant info,
# FFN processes and transforms that info.
#
# MATH:
#   FFN(x) = Linear(ReLU(Linear(x)))
#   dim: d_model → d_ff → d_model
#
# Typically d_ff = 4 * d_model (expand then compress)
# ============================================================

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Args:
        d_model : input/output dimension
        d_ff    : hidden layer dimension (usually 4 * d_model)
        dropout : dropout rate
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        # Two linear transformations with ReLU between them
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(dropout)
        self.relu     = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x : input, shape (batch, seq_len, d_model)
        Returns:
            output, same shape as input
        """
        # Expand → activate → dropout → compress
        x = self.linear1(x)   # (batch, seq_len, d_ff)
        x = self.relu(x)       # non-linearity
        x = self.dropout(x)    # regularization
        x = self.linear2(x)   # (batch, seq_len, d_model)
        return x


# ============================================================
# COMPONENT 5: TRANSFORMER BLOCK
# ============================================================
# One complete transformer layer combining:
# 1. Multi-Head Attention
# 2. Add & Norm (residual connection + layer normalization)
# 3. Feed Forward Network
# 4. Add & Norm again
#
# RESIDUAL CONNECTIONS (Add):
#   output = LayerNorm(x + sublayer(x))
#
#   WHY? They create "shortcuts" for gradients to flow
#   through during backpropagation. Without them, deep
#   networks suffer from vanishing gradient problem.
#   Think of it as: "keep the original info + add new info"
#
# LAYER NORMALIZATION:
#   Normalizes each sample's activations to mean=0, std=1
#   This stabilizes training and speeds up convergence.
# ============================================================

class TransformerBlock(nn.Module):
    """
    Single Transformer decoder block.

    Args:
        d_model   : embedding dimension
        num_heads : number of attention heads
        d_ff      : feedforward hidden dimension
        dropout   : dropout rate
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Sub-layer 1: Masked Multi-Head Self-Attention
        self.attention    = MultiHeadAttention(d_model, num_heads, dropout)

        # Sub-layer 2: Feed Forward Network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer normalization (applied after each sub-layer)
        # Normalizes to stabilize training
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x    : input, shape (batch, seq_len, d_model)
            mask : causal mask
        Returns:
            output, same shape as input
        """

        # === Sub-layer 1: Attention ===
        # 1a. Apply attention
        attn_output = self.attention(x, mask)

        # 1b. Residual connection + Layer Norm
        # "Add & Norm" from the paper diagram
        # x + attn_output = residual (skip connection)
        x = self.norm1(x + self.dropout(attn_output))

        # === Sub-layer 2: Feed Forward ===
        # 2a. Apply feed forward
        ff_output = self.feed_forward(x)

        # 2b. Residual connection + Layer Norm
        x = self.norm2(x + self.dropout(ff_output))

        return x


# ============================================================
# COMPONENT 6: COMPLETE TRANSFORMER MODEL (MiniGPT)
# ============================================================
# Decoder-only transformer (like GPT, not original encoder-decoder)
#
# WHY decoder-only?
# For language modeling (predict next token), we only need
# the decoder. The encoder is for when you have a separate
# input sequence to encode (like in translation).
# Our task: given all previous tokens, predict the next one.
# That's exactly what a decoder does.
#
# HYPERPARAMETERS (things you can tune):
#   vocab_size  : size of tokenizer vocabulary (fixed = 40479)
#   d_model     : embedding size (try 128, 256)
#   num_heads   : attention heads (must divide d_model evenly)
#   num_layers  : number of transformer blocks (try 2, 4)
#   d_ff        : feedforward size (usually 4 * d_model)
#   max_seq_len : maximum sequence length
#   dropout     : regularization strength (0.1 is standard)
# ============================================================

class MiniGPT(nn.Module):
    """
    Small decoder-only Transformer for QA generation.

    Args:
        vocab_size  : tokenizer vocabulary size
        d_model     : embedding dimension
        num_heads   : number of attention heads
        num_layers  : number of transformer blocks
        d_ff        : feedforward hidden dimension
        max_seq_len : maximum input sequence length
        dropout     : dropout rate
    """

    def __init__(self,
                 vocab_size,
                 d_model     = 256,
                 num_heads   = 8,
                 num_layers  = 4,
                 d_ff        = 1024,
                 max_seq_len = 256,
                 dropout     = 0.1):
        super().__init__()

        # Save config for reference
        self.d_model     = d_model
        self.vocab_size  = vocab_size
        self.num_layers  = num_layers

        # --- Layer 1: Token Embedding ---
        # Maps each token ID to a d_model-dimensional vector
        # The model LEARNS these vectors during training
        # padding_idx tells model to ignore PAD tokens
        self.embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=0
        )

        # --- Layer 2: Positional Encoding ---
        # Adds position information to embeddings
        self.pos_encoding = PositionalEncoding(
            d_model, max_seq_len, dropout
        )

        # --- Layer 3: Stack of Transformer Blocks ---
        # nn.ModuleList properly registers all blocks as submodules
        # so their parameters are included in model.parameters()
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # --- Layer 4: Final Layer Normalization ---
        self.norm = nn.LayerNorm(d_model)

        # --- Layer 5: Output Projection ---
        # Maps d_model vectors → vocabulary logits
        # logits = raw scores for each possible next token
        # (softmax converts these to probabilities)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights properly
        # This helps the model train faster and more stably
        self._initialize_weights()

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        print(f"MiniGPT initialized:")
        print(f"  Vocab size   : {vocab_size}")
        print(f"  d_model      : {d_model}")
        print(f"  Num heads    : {num_heads}")
        print(f"  Num layers   : {num_layers}")
        print(f"  d_ff         : {d_ff}")
        print(f"  Total params : {total_params:,}")

    def _initialize_weights(self):
        """
        Initialize model weights using standard techniques.

        WHY? Random initialization can cause slow training.
        Xavier/Glorot initialization sets weights to values
        that keep gradients from vanishing or exploding.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                # Small normal distribution for embeddings
                nn.init.normal_(module.weight, mean=0, std=0.01)

    def create_causal_mask(self, seq_len, device):
        """
        Create causal (look-ahead) mask.

        Prevents position i from attending to positions > i.
        This ensures autoregressive property:
        each token can only see tokens before it.

        Returns lower triangular matrix:
            pos 0: [1, 0, 0, 0]   can see: only itself
            pos 1: [1, 1, 0, 0]   can see: pos 0, 1
            pos 2: [1, 1, 1, 0]   can see: pos 0, 1, 2
            pos 3: [1, 1, 1, 1]   can see: all positions

        Args:
            seq_len : length of sequence
            device  : cpu or cuda (must match input tensor)

        Returns:
            mask : shape (seq_len, seq_len)
        """
        # torch.tril = lower triangular part of matrix
        # ones matrix with lower triangle = 1, upper = 0
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device)
        )
        return mask

    def forward(self, x):
        """
        Forward pass through the full model.

        Args:
            x : token IDs, shape (batch_size, seq_len)

        Returns:
            logits : shape (batch_size, seq_len, vocab_size)
                     raw scores for each possible next token
                     at each position
        """
        batch_size, seq_len = x.shape

        # Create causal mask for this sequence length
        # Must be on same device as input
        mask = self.create_causal_mask(seq_len, x.device)

        # Step 1: Token Embedding
        # (batch, seq_len) → (batch, seq_len, d_model)
        # Scale by sqrt(d_model) as in original paper
        # This prevents embeddings from being too small
        # relative to positional encodings
        x = self.embedding(x) * math.sqrt(self.d_model)

        # Step 2: Add Positional Encoding
        # Still shape (batch, seq_len, d_model)
        x = self.pos_encoding(x)

        # Step 3: Pass through all Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Step 4: Final Layer Normalization
        x = self.norm(x)

        # Step 5: Project to vocabulary size
        # (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        logits = self.output_projection(x)

        return logits


# ============================================================
# QUICK TEST - verify model works before training
# ============================================================
if __name__ == "__main__":

    print("Testing model.py...")
    print("="*60)

    # Use CPU for testing
    device = torch.device('cpu')

    # Create a small model for testing
    model = MiniGPT(
        vocab_size  = 40479,
        d_model     = 128,    # Small for quick test
        num_heads   = 4,
        num_layers  = 2,
        d_ff        = 512,
        max_seq_len = 128,
        dropout     = 0.1
    ).to(device)

    # Create fake input batch
    # Shape: (batch_size=2, seq_len=64)
    # Random token IDs between 0 and vocab_size
    fake_input = torch.randint(0, 40479, (2, 64))

    print("\nRunning forward pass...")
    output = model(fake_input)

    print(f"\nInput shape  : {fake_input.shape}")
    print(f"Output shape : {output.shape}")
    print(f"Expected     : (2, 64, 40479)")

    # Verify output shape is correct
    assert output.shape == (2, 64, 40479), "Shape mismatch!"

    # Test causal mask
    mask = model.create_causal_mask(5, device)
    print(f"\nCausal mask (5x5):")
    print(mask)
    print("(Upper triangle should be all zeros)")

    print("\nmodel.py is working correctly!")

# train.py
# PURPOSE  : Training loop for MiniGPT QA model
# Author   : Saad Ali (MSDS25066)
#
# TRAINING FLOW:
#   For each epoch:
#     1. Loop through all training batches
#     2. Forward pass → get predictions
#     3. Compute loss (CrossEntropy)
#     4. Backward pass → compute gradients
#     5. Update weights (optimizer step)
#     6. Evaluate on validation set
#     7. Save best model (checkpointing)
#     8. Check early stopping condition
#   End
#
# KEY CONCEPTS:
#   Loss        : CrossEntropyLoss - measures prediction error
#   Perplexity  : exp(loss) - how "confused" the model is
#   BLEU        : measures similarity to reference answers
#   Early Stop  : stop when validation loss stops improving
#   Checkpoint  : save model whenever validation loss improves

import os
import math
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')           # non-interactive backend for saving
import matplotlib.pyplot as plt
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download NLTK data needed for BLEU score
nltk.download('punkt', quiet=True)

# Make sure output folders exist
os.makedirs('weights', exist_ok=True)
os.makedirs('graphs',  exist_ok=True)


# ============================================================
# METRIC 1: PERPLEXITY
# ============================================================
# Perplexity = exp(cross_entropy_loss)
#
# INTUITION:
#   Perplexity of N means the model is as confused as if it
#   had to pick uniformly from N equally likely options.
#
#   Perplexity = 1    → perfect (model always right)
#   Perplexity = 100  → model equally confused between 100 words
#   Perplexity = 40479 → model is completely random (vocab size)
#
# GOOD VALUES for a small language model:
#   < 50   → excellent for a tiny model
#   50-200 → decent
#   > 500  → model is not learning well
# ============================================================

def compute_perplexity(loss_value):
    """
    Compute perplexity from cross-entropy loss.

    Args:
        loss_value : float, average cross-entropy loss

    Returns:
        perplexity : float
    """
    # Clamp loss to prevent overflow in exp()
    # exp(20) is already huge, anything above is meaningless
    loss_value = min(loss_value, 20.0)
    return math.exp(loss_value)


# ============================================================
# METRIC 2: BLEU SCORE
# ============================================================
# BLEU = Bilingual Evaluation Understudy
#
# INTUITION:
#   Measures how many n-grams (word sequences) in the
#   generated answer match the reference answer.
#
#   BLEU of 1.0 = perfect match
#   BLEU of 0.0 = no overlap at all
#   BLEU > 0.3  = decent for open-ended generation
#
# Example:
#   Reference : "use pandas read csv function"
#   Generated : "you can use pandas to read csv"
#   Overlap   : "use", "pandas", "read", "csv" → decent BLEU
# ============================================================

def compute_bleu(references, hypotheses):
    """
    Compute average BLEU score over a list of samples.

    Args:
        references  : list of reference strings (ground truth)
        hypotheses  : list of generated strings (model output)

    Returns:
        avg_bleu : float between 0 and 1
    """
    smoothing = SmoothingFunction().method1
    scores = []

    for ref, hyp in zip(references, hypotheses):
        # Tokenize by splitting on spaces
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()

        # Need at least 1 token to compute score
        if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
            scores.append(0.0)
            continue

        score = sentence_bleu(
            [ref_tokens],           # reference (list of list)
            hyp_tokens,             # hypothesis (list)
            smoothing_function=smoothing
        )
        scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


# ============================================================
# TRAINING FUNCTION: ONE EPOCH
# ============================================================
# One epoch = one complete pass through ALL training data
#
# For each batch:
#   1. Move data to GPU/CPU
#   2. Forward pass: model(input) → logits
#   3. Compute loss: CrossEntropy(logits, targets)
#   4. Zero gradients (IMPORTANT: must clear before backward)
#   5. Backward pass: compute gradients via chain rule
#   6. Clip gradients: prevent exploding gradients
#   7. Optimizer step: update weights using gradients
# ============================================================

def train_one_epoch(model, loader, optimizer,
                    criterion, device, epoch_num):
    """
    Train model for one complete epoch.

    Args:
        model     : MiniGPT model
        loader    : training DataLoader
        optimizer : AdamW optimizer
        criterion : CrossEntropyLoss
        device    : 'cuda' or 'cpu'
        epoch_num : current epoch number (for display)

    Returns:
        avg_loss  : average loss over all batches
        perplexity: exp(avg_loss)
    """

    # Set model to training mode
    # This ENABLES dropout (regularization during training)
    model.train()

    total_loss   = 0.0
    total_batches = 0

    # tqdm = progress bar so you can see training progress
    progress_bar = tqdm(
        loader,
        desc=f"  Epoch {epoch_num} [Train]",
        leave=False
    )

    for batch_idx, (inputs, targets) in enumerate(progress_bar):

        # Move tensors to device (GPU if available)
        inputs  = inputs.to(device)   # shape: (batch, seq_len-1)
        targets = targets.to(device)  # shape: (batch, seq_len-1)

        # ── FORWARD PASS ──────────────────────────────────
        # Feed input through model to get predictions
        logits = model(inputs)
        # logits shape: (batch, seq_len-1, vocab_size)

        # ── COMPUTE LOSS ──────────────────────────────────
        # CrossEntropyLoss expects:
        #   input  shape: (N, num_classes)
        #   target shape: (N,)
        # So we flatten batch and sequence dimensions together
        batch_size, seq_len, vocab_size = logits.shape

        loss = criterion(
            logits.reshape(-1, vocab_size),  # (batch*seq_len, vocab_size)
            targets.reshape(-1)              # (batch*seq_len,)
        )

        # ── BACKWARD PASS ─────────────────────────────────
        # Step 1: Clear gradients from previous batch
        # WHY? PyTorch accumulates gradients by default
        # If we don't clear, gradients add up incorrectly
        optimizer.zero_grad()

        # Step 2: Compute gradients via backpropagation
        # This computes dLoss/dW for every weight W in the model
        loss.backward()

        # Step 3: Gradient clipping
        # WHY? Sometimes gradients explode (become huge)
        # making training unstable. We clip to max norm of 1.0
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        # Step 4: Update weights using computed gradients
        optimizer.step()

        # Track loss
        total_loss    += loss.item()
        total_batches += 1

        # Update progress bar with current loss
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss   = total_loss / total_batches
    perplexity = compute_perplexity(avg_loss)

    return avg_loss, perplexity


# ============================================================
# EVALUATION FUNCTION
# ============================================================
# Same as training but:
# - model.eval() DISABLES dropout (deterministic predictions)
# - torch.no_grad() skips gradient computation (faster + less memory)
# - We do NOT call optimizer.step() (no weight updates)
# ============================================================

def evaluate(model, loader, criterion, device, tokenizer=None):
    """
    Evaluate model on validation or test set.

    Args:
        model     : MiniGPT model
        loader    : val or test DataLoader
        criterion : CrossEntropyLoss
        device    : 'cuda' or 'cpu'
        tokenizer : optional, needed for BLEU computation

    Returns:
        avg_loss   : float
        perplexity : float
        bleu_score : float (0 if tokenizer not provided)
    """

    # Evaluation mode: disables dropout for deterministic output
    model.eval()

    total_loss    = 0.0
    total_batches = 0
    all_references  = []
    all_hypotheses  = []

    # no_grad: don't compute gradients (saves memory and time)
    # We don't need gradients during evaluation
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="  Evaluating", leave=False):

            inputs  = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(inputs)

            batch_size, seq_len, vocab_size = logits.shape

            # Compute loss
            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            total_loss    += loss.item()
            total_batches += 1

            # Collect predictions for BLEU score
            # Only compute BLEU if tokenizer is provided
            if tokenizer is not None:
                # Get predicted token IDs (argmax over vocab dimension)
                predicted_ids = logits.argmax(dim=-1)  # (batch, seq_len)

                for i in range(inputs.size(0)):
                    # Decode predicted and target sequences to text
                    pred_text = tokenizer.decode(
                        predicted_ids[i].cpu(),
                        skip_special_tokens=True
                    )
                    ref_text = tokenizer.decode(
                        targets[i].cpu(),
                        skip_special_tokens=True
                    )
                    all_hypotheses.append(pred_text)
                    all_references.append(ref_text)

    avg_loss   = total_loss / total_batches
    perplexity = compute_perplexity(avg_loss)

    # Compute BLEU if we collected predictions
    bleu = 0.0
    if all_references and all_hypotheses:
        bleu = compute_bleu(all_references, all_hypotheses)

    return avg_loss, perplexity, bleu


# ============================================================
# VISUALIZATION: PLOT TRAINING CURVES
# ============================================================

def plot_training_curves(history, save_dir='graphs/'):
    """
    Plot and save training/validation loss and perplexity curves.

    Args:
        history  : dict with keys train_loss, val_loss,
                   train_ppl, val_ppl, val_bleu
        save_dir : folder to save graph images
    """
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # ── Figure 1: Loss Curves ─────────────────────────────
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-o',
             label='Train Loss',      linewidth=2)
    plt.plot(epochs, history['val_loss'],   'r-o',
             label='Validation Loss', linewidth=2)
    plt.title('Loss per Epoch',  fontsize=13)
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ── Figure 2: Perplexity Curves ───────────────────────
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_ppl'], 'b-o',
             label='Train PPL',      linewidth=2)
    plt.plot(epochs, history['val_ppl'],   'r-o',
             label='Validation PPL', linewidth=2)
    plt.title('Perplexity per Epoch', fontsize=13)
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # ── Figure 3: BLEU Score ──────────────────────────────
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_bleu'], 'g-o',
             label='Validation BLEU', linewidth=2)
    plt.title('BLEU Score per Epoch', fontsize=13)
    plt.xlabel('Epoch')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Graphs saved → {save_path}")


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model(model, train_loader, val_loader,
                tokenizer, config, device):
    """
    Full training pipeline with all callbacks.

    Callbacks implemented:
    1. EarlyStopping    : stop if val_loss doesn't improve
    2. Checkpointing    : save best model weights
    3. LR Scheduler     : CosineAnnealingLR

    Args:
        model        : MiniGPT model
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        tokenizer    : for BLEU computation
        config       : dict of hyperparameters
        device       : 'cuda' or 'cpu'

    Returns:
        history : dict of recorded metrics per epoch
    """

    # ── Loss Function ─────────────────────────────────────
    # CrossEntropyLoss:
    # - Measures how wrong the model's predictions are
    # - ignore_index: don't compute loss for PAD tokens
    #   (they're not real content, shouldn't affect training)
    criterion = nn.CrossEntropyLoss(
        ignore_index=config['pad_token_id']
    )

    # ── Optimizer: AdamW ──────────────────────────────────
    # AdamW = Adam with Weight Decay (better than plain Adam)
    # weight_decay = L2 regularization (prevents overfitting)
    # lr = learning rate (how big each weight update step is)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config['learning_rate'],
        weight_decay = 0.01
    )

    # ── LR Scheduler: CosineAnnealingLR ───────────────────
    # Gradually decreases learning rate following cosine curve
    #
    # WHY? Large LR at start = fast learning
    #      Small LR at end   = fine-tuning, stable convergence
    #
    # Cosine shape:    high → slowly decreases → low
    # (smoother than StepLR which drops suddenly)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = config['num_epochs'],  # period of cosine
        eta_min= 1e-6                   # minimum LR
    )

    # ── Tracking ──────────────────────────────────────────
    history = {
        'train_loss': [],
        'val_loss'  : [],
        'train_ppl' : [],
        'val_ppl'   : [],
        'val_bleu'  : []
    }

    # ── Early Stopping State ──────────────────────────────
    # patience = how many epochs to wait for improvement
    # before stopping training
    best_val_loss    = float('inf')
    patience_counter = 0

    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"  Epochs        : {config['num_epochs']}")
    print(f"  Learning Rate : {config['learning_rate']}")
    print(f"  Batch Size    : {config['batch_size']}")
    print(f"  Device        : {device}")
    print(f"  Early Stop    : patience={config['patience']}")
    print("="*60)

    for epoch in range(1, config['num_epochs'] + 1):

        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-"*40)

        # ── Train ─────────────────────────────────────────
        train_loss, train_ppl = train_one_epoch(
            model, train_loader, optimizer,
            criterion, device, epoch
        )

        # ── Validate ──────────────────────────────────────
        val_loss, val_ppl, val_bleu = evaluate(
            model, val_loader, criterion, device, tokenizer
        )

        # ── Update Scheduler ──────────────────────────────
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Record History ────────────────────────────────
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_ppl'].append(train_ppl)
        history['val_ppl'].append(val_ppl)
        history['val_bleu'].append(val_bleu)

        # ── Print Epoch Summary ───────────────────────────
        print(f"  Train  → Loss: {train_loss:.4f} | PPL: {train_ppl:.2f}")
        print(f"  Val    → Loss: {val_loss:.4f}   | PPL: {val_ppl:.2f} | BLEU: {val_bleu:.4f}")
        print(f"  LR     : {current_lr:.6f}")

        # ── Checkpointing ─────────────────────────────────
        # Save model whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0

            # Save full checkpoint
            torch.save({
                'epoch'               : epoch,
                'model_state_dict'    : model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss'            : val_loss,
                'val_ppl'             : val_ppl,
                'val_bleu'            : val_bleu,
                'config'              : config
            }, 'weights/best_model.pt')

            print(f"  ✓ Saved best model  (val_loss={val_loss:.4f})")

        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{config['patience']}")

        # ── Early Stopping Check ──────────────────────────
        # Stop training if no improvement for 'patience' epochs
        if patience_counter >= config['patience']:
            print(f"\n  Early stopping triggered at epoch {epoch}")
            print(f"  Best val_loss was: {best_val_loss:.4f}")
            break

    # ── Save Final Graphs ─────────────────────────────────
    print("\nGenerating training curves...")
    plot_training_curves(history)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Best Val PPL  : {compute_perplexity(best_val_loss):.2f}")
    print("="*60)

    return history


# ============================================================
# QUICK TEST - verify train.py works before real training
# ============================================================
if __name__ == "__main__":

    print("Testing train.py with tiny data...")
    print("="*60)

    from data_utils import prepare_data
    from model import MiniGPT

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load tiny dataset for quick test
    train_loader, val_loader, test_loader, tokenizer = prepare_data(
        questions_path = 'dataset/Questions.csv',
        answers_path   = 'dataset/Answers.csv',
        num_samples    = 200,    # tiny for testing
        max_length     = 128,
        batch_size     = 8
    )

    # Small model for quick test
    model = MiniGPT(
        vocab_size  = len(tokenizer),
        d_model     = 128,
        num_heads   = 4,
        num_layers  = 2,
        d_ff        = 512,
        max_seq_len = 128,
        dropout     = 0.1
    ).to(device)

    # Training config
    config = {
        'num_epochs'    : 3,      # only 3 epochs for test
        'learning_rate' : 3e-4,
        'batch_size'    : 8,
        'patience'      : 3,
        'pad_token_id'  : tokenizer.pad_token_id
    }

    # Run training
    history = train_model(
        model, train_loader, val_loader,
        tokenizer, config, device
    )

    print("\nFinal history keys:", list(history.keys()))
    print("Epochs recorded   :", len(history['train_loss']))
    print("\ntrain.py test complete!")

# test.py
# PURPOSE  : Evaluate the trained model on the test set
# Author   : Saad Ali (MSDS25066)
#
# This file:
# 1. Loads the best saved model from weights/best_model.pt
# 2. Runs evaluation on the held-out test set
# 3. Reports Loss, Perplexity, and BLEU score
# 4. Saves sample predictions to a text file for the report

import os
import torch
import torch.nn as nn

from data_utils import prepare_data
from model import MiniGPT
from train import evaluate, compute_perplexity


def load_best_model(checkpoint_path, device):
    """
    Load the best saved model from a checkpoint file.

    A checkpoint contains:
    - model_state_dict  : all learned weights
    - optimizer_state_dict : optimizer state
    - config            : hyperparameters used during training
    - val_loss          : best validation loss achieved

    Args:
        checkpoint_path : path to .pt file
        device          : 'cuda' or 'cpu'

    Returns:
        model     : loaded MiniGPT model (eval mode)
        config    : training config dict
        tokenizer : same tokenizer used during training
    """

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load checkpoint dictionary
    # map_location ensures it loads on correct device
    # (model might have been saved on GPU, loading on CPU)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']

    print(f"  Checkpoint from epoch : {checkpoint['epoch']}")
    print(f"  Best val loss         : {checkpoint['val_loss']:.4f}")
    print(f"  Best val PPL          : {checkpoint['val_ppl']:.2f}")

    return checkpoint, config


def run_test_evaluation(checkpoint_path='weights/best_model.pt',
                        questions_path='dataset/Questions.csv',
                        answers_path='dataset/Answers.csv',
                        num_samples=5000):
    """
    Full test evaluation pipeline.

    Steps:
    1. Load data (same splits as training)
    2. Load best model
    3. Evaluate on test set
    4. Print and save results

    Args:
        checkpoint_path : path to saved model
        questions_path  : path to Questions.csv
        answers_path    : path to Answers.csv
        num_samples     : must match training num_samples!
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Step 1: Load Data ──────────────────────────────────
    # IMPORTANT: use same num_samples and random_state as training
    # This guarantees the test set has no overlap with training data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, tokenizer = prepare_data(
        questions_path = questions_path,
        answers_path   = answers_path,
        num_samples    = num_samples,
        max_length     = 128,
        batch_size     = 16
    )

    # ── Step 2: Load Best Model ────────────────────────────
    print("\nLoading best model...")
    checkpoint, config = load_best_model(checkpoint_path, device)

    # Rebuild model with same architecture as training
    model = MiniGPT(
        vocab_size  = len(tokenizer),
        d_model     = config.get('d_model',     128),
        num_heads   = config.get('num_heads',   4),
        num_layers  = config.get('num_layers',  2),
        d_ff        = config.get('d_ff',        512),
        max_seq_len = config.get('max_seq_len', 128),
        dropout     = 0.0    # No dropout during evaluation!
    ).to(device)

    # Load the saved weights into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode (disables dropout)
    model.eval()
    print("  Model loaded successfully!")

    # ── Step 3: Evaluate on Test Set ──────────────────────
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id
    )

    print("\nRunning test evaluation...")
    test_loss, test_ppl, test_bleu = evaluate(
        model, test_loader, criterion, device, tokenizer
    )

    # ── Step 4: Print Results ─────────────────────────────
    print("\n" + "="*50)
    print("TEST SET RESULTS")
    print("="*50)
    print(f"  Test Loss       : {test_loss:.4f}")
    print(f"  Test Perplexity : {test_ppl:.2f}")
    print(f"  Test BLEU Score : {test_bleu:.4f}")
    print("="*50)

    # ── Step 5: Save Sample Predictions ───────────────────
    # Generate a few sample predictions to include in report
    save_sample_predictions(
        model, test_loader, tokenizer, device, num_samples=5
    )

    return test_loss, test_ppl, test_bleu


def save_sample_predictions(model, loader, tokenizer,
                             device, num_samples=5):
    """
    Generate and save sample predictions for the report.

    Takes the first batch from the loader,
    decodes both input and predicted tokens,
    and saves them to a text file.

    Args:
        model       : loaded MiniGPT model
        loader      : test DataLoader
        tokenizer   : for decoding token IDs
        device      : cpu or cuda
        num_samples : how many examples to save
    """

    model.eval()
    os.makedirs('graphs', exist_ok=True)

    # Get one batch
    inputs, targets = next(iter(loader))
    inputs  = inputs.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        logits = model(inputs)
        # Get predicted token at each position
        predicted_ids = logits.argmax(dim=-1)  # (batch, seq_len)

    # Save to file
    save_path = 'graphs/sample_predictions.txt'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("SAMPLE PREDICTIONS FROM TEST SET\n")
        f.write("="*60 + "\n\n")

        for i in range(min(num_samples, inputs.size(0))):

            # Decode input (question part)
            input_text = tokenizer.decode(
                inputs[i].cpu(),
                skip_special_tokens=True
            )

            # Decode target (real answer)
            target_text = tokenizer.decode(
                targets[i].cpu(),
                skip_special_tokens=True
            )

            # Decode prediction (model's answer)
            pred_text = tokenizer.decode(
                predicted_ids[i].cpu(),
                skip_special_tokens=True
            )

            f.write(f"--- Sample {i+1} ---\n")
            f.write(f"INPUT    : {input_text[:300]}\n\n")
            f.write(f"TARGET   : {target_text[:300]}\n\n")
            f.write(f"PREDICTED: {pred_text[:300]}\n\n")
            f.write("-"*60 + "\n\n")

    print(f"\nSample predictions saved → {save_path}")
    print("(Include these in your report!)")


# ============================================================
# RUN TEST
# ============================================================
if __name__ == "__main__":

    print("="*60)
    print("TEST EVALUATION")
    print("="*60)

    # Check if best model exists
    if not os.path.exists('weights/best_model.pt'):
        print("ERROR: No saved model found at weights/best_model.pt")
        print("Please run train.py first to train and save a model.")
    else:
        run_test_evaluation(
            checkpoint_path = 'weights/best_model.pt',
            questions_path  = 'dataset/Questions.csv',
            answers_path    = 'dataset/Answers.csv',
            num_samples     = 200   # match what train.py used
        )

# test_eval.py
# PURPOSE  : Interactive Q&A chatbot using trained MiniGPT model
# Author   : Saad Ali (MSDS25066)
# Usage    : python test_eval.py
#
# HOW ANSWER GENERATION WORKS:
#
#   We use TOP-K SAMPLING with TEMPERATURE:
#
#   At each step:
#   1. Feed all tokens so far into the model
#   2. Get logits (scores) for every possible next token
#   3. Apply temperature (scale the scores)
#   4. Keep only top-k highest scoring tokens
#   5. Convert to probabilities (softmax)
#   6. SAMPLE from those probabilities (not just pick the best)
#   7. Append sampled token to sequence
#   8. Repeat until <END> token or max_length reached
#
#   WHY SAMPLING instead of always picking the best token?
#   "Greedy" decoding (always pick #1) produces repetitive,
#   boring text. Sampling adds natural variety.
#
#   TEMPERATURE controls creativity:
#   < 1.0 = more focused/conservative (safer answers)
#   > 1.0 = more random/creative (sometimes nonsensical)
#   = 1.0 = neutral (raw model probabilities)
#
#   TOP-K limits choices to only k most likely tokens,
#   preventing the model from picking very unlikely words.

import os
import sys
import torch
from transformers import AutoTokenizer
from model import MiniGPT


# ============================================================
# STEP 1: LOAD MODEL FROM CHECKPOINT
# ============================================================

def load_model_for_inference(checkpoint_path='weights/best_model.pt',
                              device=None):
    """
    Load the best trained model for answer generation.

    Args:
        checkpoint_path : path to saved .pt file
        device          : torch device (auto-detected if None)

    Returns:
        model     : MiniGPT in eval mode
        tokenizer : same tokenizer used in training
        config    : model hyperparameters
    """

    # Auto-detect device
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    print(f"Loading model on: {device}")

    # Check file exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: No model found at '{checkpoint_path}'")
        print("Please run train.py first!")
        sys.exit(1)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config     = checkpoint['config']

    print(f"  Trained for epochs : {checkpoint['epoch']}")
    print(f"  Best val loss      : {checkpoint['val_loss']:.4f}")
    print(f"  Best val PPL       : {checkpoint['val_ppl']:.2f}")

    # Load tokenizer (must be same as training!)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "openai-community/openai-gpt"
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Rebuild model architecture
    model = MiniGPT(
        vocab_size  = len(tokenizer),
        d_model     = config.get('d_model',     128),
        num_heads   = config.get('num_heads',   4),
        num_layers  = config.get('num_layers',  2),
        d_ff        = config.get('d_ff',        512),
        max_seq_len = config.get('max_seq_len', 128),
        dropout     = 0.0    # No dropout during inference!
    ).to(device)

    # Load saved weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode
    model.eval()

    print("  Model ready for inference!")
    return model, tokenizer, config, device


# ============================================================
# STEP 2: TOP-K SAMPLING
# ============================================================

def top_k_sampling(logits, k=50, temperature=0.8):
    """
    Apply temperature scaling and top-k filtering,
    then sample the next token.

    Args:
        logits      : raw model output, shape (vocab_size,)
        k           : keep only top-k tokens
        temperature : controls randomness

    Returns:
        next_token_id : int, the sampled token
    """

    # Step 1: Apply temperature
    # Dividing by temperature < 1 makes distribution sharper
    # (high probability tokens become even more likely)
    # Dividing by temperature > 1 makes it flatter (more random)
    logits = logits / temperature

    # Step 2: Top-k filtering
    # Find the k-th largest value
    # tokens below this threshold get set to -infinity
    if k > 0:
        # Get top-k values and their indices
        top_k_values, _ = torch.topk(logits, min(k, logits.size(-1)))

        # Find the minimum value among top-k
        min_top_k = top_k_values[-1]

        # Set all tokens below top-k threshold to -infinity
        # After softmax, -inf → probability 0
        logits[logits < min_top_k] = float('-inf')

    # Step 3: Convert to probabilities
    probabilities = torch.softmax(logits, dim=-1)

    # Step 4: Sample one token from the probability distribution
    # multinomial = weighted random sampling
    next_token = torch.multinomial(probabilities, num_samples=1)

    return next_token.item()


# ============================================================
# STEP 3: ANSWER GENERATION
# ============================================================

def generate_answer(model, tokenizer, question_title,
                    question_body="", max_new_tokens=150,
                    temperature=0.8, top_k=50, device=None):
    """
    Generate an answer for a given question.

    The model generates ONE TOKEN AT A TIME:
    - Start with: <START> question [SEP]
    - Generate token 1 → append → generate token 2 → ...
    - Stop when <END> token appears or max_new_tokens reached

    Args:
        model           : trained MiniGPT
        tokenizer       : matching tokenizer
        question_title  : question title string
        question_body   : question description (optional)
        max_new_tokens  : maximum tokens to generate
        temperature     : creativity control (0.7-1.0 recommended)
        top_k           : vocabulary size limit per step
        device          : cpu or cuda

    Returns:
        answer : generated answer string
    """

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # ── Format the prompt ─────────────────────────────────
    # Same format as training data!
    # Model learned this pattern, so we must use same format
    import re
    from bs4 import BeautifulSoup

    def quick_clean(text):
        """Quick clean for inference input"""
        if not isinstance(text, str):
            return ""
        text = BeautifulSoup(text, 'html.parser').get_text()
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s.,?!]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    clean_title = quick_clean(question_title)
    clean_body  = quick_clean(question_body)

    # Build prompt: everything before the answer
    if clean_body:
        prompt = (f"<saadali_msds25066> "
                  f"{clean_title} . {clean_body} "
                  f"[sep]")
    else:
        prompt = (f"<saadali_msds25066> "
                  f"{clean_title} "
                  f"[sep]")

    # ── Tokenize the prompt ───────────────────────────────
    input_ids = tokenizer.encode(
        prompt,
        return_tensors='pt'
    ).to(device)

    # Keep track of max context length
    max_seq_len = 128
    generated_tokens = []

    # ── Generate tokens one by one ────────────────────────
    with torch.no_grad():

        for step in range(max_new_tokens):

            # If sequence too long, truncate from the LEFT
            # (keep most recent context)
            current_input = input_ids
            if current_input.size(1) >= max_seq_len:
                current_input = current_input[:, -(max_seq_len-1):]

            # Forward pass → get logits for all positions
            logits = model(current_input)

            # We only care about the LAST position's logits
            # (predicting what comes AFTER the current sequence)
            next_token_logits = logits[0, -1, :]  # shape: (vocab_size,)

            # Sample next token using top-k sampling
            next_token_id = top_k_sampling(
                next_token_logits.clone(),
                k           = top_k,
                temperature = temperature
            )

            # Stop conditions:
            # 1. Generated the end token
            end_token = tokenizer.encode("</saadali_msds25066>")
            if next_token_id in end_token:
                break

            # 2. Generated EOS token
            if next_token_id == tokenizer.eos_token_id:
                break

            # Append new token to sequence
            generated_tokens.append(next_token_id)
            new_token_tensor = torch.tensor(
                [[next_token_id]], device=device
            )
            input_ids = torch.cat([input_ids, new_token_tensor], dim=1)

    # ── Decode generated tokens to text ───────────────────
    if generated_tokens:
        answer = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
    else:
        answer = "[Model generated no tokens. Try different temperature or train longer.]"

    return answer.strip()


# ============================================================
# STEP 4: INTERACTIVE LOOP
# ============================================================

def interactive_qa(checkpoint_path='weights/best_model.pt'):
    """
    Run interactive question-answering session.

    User types a question → model generates an answer.
    Type 'quit' or 'exit' to stop.
    Type 'help' to see options.
    """

    # Load model
    model, tokenizer, config, device = load_model_for_inference(
        checkpoint_path
    )

    # Default generation settings
    temperature = 0.8
    top_k       = 50

    print("\n" + "="*60)
    print("  STACK OVERFLOW PYTHON Q&A BOT")
    print("  Powered by MiniGPT (Saad Ali - MSDS25066)")
    print("="*60)
    print("  Ask Python programming questions!")
    print("  Commands:")
    print("    'quit'  → exit")
    print("    'temp X'→ set temperature (e.g. 'temp 0.7')")
    print("    'topk X'→ set top-k (e.g. 'topk 30')")
    print("="*60 + "\n")

    while True:

        # Get user input
        try:
            user_input = input("Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        # Handle empty input
        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        elif user_input.lower() == 'help':
            print("\nCommands:")
            print("  quit       → exit the program")
            print("  temp 0.7   → set temperature (lower=focused)")
            print("  topk 30    → set top-k (lower=conservative)")
            print(f"  Current: temperature={temperature}, top_k={top_k}\n")
            continue

        elif user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input.split()[1])
                print(f"  Temperature set to {temperature}")
            except:
                print("  Usage: temp 0.8")
            continue

        elif user_input.lower().startswith('topk '):
            try:
                top_k = int(user_input.split()[1])
                print(f"  Top-k set to {top_k}")
            except:
                print("  Usage: topk 50")
            continue

        # Generate answer
        print("\nGenerating answer...\n")

        answer = generate_answer(
            model       = model,
            tokenizer   = tokenizer,
            question_title = user_input,
            max_new_tokens = 150,
            temperature    = temperature,
            top_k          = top_k,
            device         = device
        )

        print(f"Answer: {answer}")
        print("\n" + "-"*60 + "\n")


# ============================================================
# BATCH EVALUATION MODE
# ============================================================
# Run this to generate answers for multiple questions at once
# Useful for generating examples for your report

def batch_generate_examples(checkpoint_path='weights/best_model.pt'):
    """
    Generate answers for a fixed set of test questions.
    Saves results to a file for the report.
    """

    model, tokenizer, config, device = load_model_for_inference(
        checkpoint_path
    )

    # Sample Python questions to test
    test_questions = [
        "How do I read a CSV file in Python?",
        "How to reverse a list in Python?",
        "What is the difference between list and tuple?",
        "How do I handle exceptions in Python?",
        "How to sort a dictionary by value in Python?",
    ]

    print("\n" + "="*60)
    print("BATCH GENERATION EXAMPLES")
    print("="*60)

    results = []

    for i, question in enumerate(test_questions, 1):
        print(f"\nQ{i}: {question}")

        answer = generate_answer(
            model          = model,
            tokenizer      = tokenizer,
            question_title = question,
            max_new_tokens = 100,
            temperature    = 0.8,
            top_k          = 50,
            device         = device
        )

        print(f"A{i}: {answer}")
        results.append({'question': question, 'answer': answer})

    # Save to file for report
    os.makedirs('graphs', exist_ok=True)
    save_path = 'graphs/generated_examples.txt'

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("GENERATED QA EXAMPLES\n")
        f.write("Model: MiniGPT - Saad Ali (MSDS25066)\n")
        f.write("="*60 + "\n\n")

        for i, item in enumerate(results, 1):
            f.write(f"Question {i}: {item['question']}\n")
            f.write(f"Answer   {i}: {item['answer']}\n\n")
            f.write("-"*60 + "\n\n")

    print(f"\nExamples saved → {save_path}")
    return results


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='MiniGPT QA Bot - Saad Ali MSDS25066'
    )
    parser.add_argument(
        '--mode',
        choices=['interactive', 'batch'],
        default='interactive',
        help='interactive = chat mode, batch = generate examples'
    )
    parser.add_argument(
        '--checkpoint',
        default='weights/best_model.pt',
        help='path to model checkpoint'
    )

    args = parser.parse_args()

    if args.mode == 'interactive':
        interactive_qa(args.checkpoint)
    else:
        batch_generate_examples(args.checkpoint)

# msds25066_04_task1.py
# PURPOSE  : Main training pipeline - Assignment 4
# Author   : Saad Ali (MSDS25066)
# Course   : Deep Learning - Spring 2026
#
# This file runs the complete pipeline:
#   1. Load and preprocess data
#   2. Build transformer model
#   3. Train with proper settings
#   4. Evaluate on test set
#   5. Save graphs and results
#
# Usage:
#   python msds25066_04_task1.py
#   python msds25066_04_task1.py --samples 3000 --epochs 15
#   python msds25066_04_task1.py --samples 5000 --epochs 20 --d_model 256

import os
import sys
import argparse
import torch
from data_utils import prepare_data
from model      import MiniGPT
from train      import train_model, compute_perplexity
from test       import run_test_evaluation


def parse_arguments():
    """
    Parse command line arguments.
    This makes the script flexible — you can change settings
    without editing the code directly.

    Example:
        python msds25066_04_task1.py --samples 5000 --epochs 20
    """
    parser = argparse.ArgumentParser(
        description='MiniGPT Training Pipeline - Saad Ali MSDS25066'
    )

    # Data arguments
    parser.add_argument('--questions', type=str,
                        default='dataset/Questions.csv',
                        help='Path to Questions.csv')
    parser.add_argument('--answers', type=str,
                        default='dataset/Answers.csv',
                        help='Path to Answers.csv')
    parser.add_argument('--samples', type=int,
                        default=5000,
                        help='Number of QA samples to use')
    parser.add_argument('--max_length', type=int,
                        default=128,
                        help='Maximum sequence length')

    # Model arguments
    parser.add_argument('--d_model', type=int,
                        default=256,
                        help='Embedding dimension')
    parser.add_argument('--num_heads', type=int,
                        default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int,
                        default=4,
                        help='Number of transformer blocks')
    parser.add_argument('--d_ff', type=int,
                        default=1024,
                        help='Feedforward hidden dimension')
    parser.add_argument('--dropout', type=float,
                        default=0.1,
                        help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int,
                        default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float,
                        default=3e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int,
                        default=5,
                        help='Early stopping patience')

    return parser.parse_args()


def print_config(args, device):
    """Print all configuration settings clearly."""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"  Device        : {device}")
    print(f"  Samples       : {args.samples}")
    print(f"  Max Length    : {args.max_length}")
    print(f"  Batch Size    : {args.batch_size}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Learning Rate : {args.lr}")
    print(f"  Patience      : {args.patience}")
    print("-"*60)
    print(f"  d_model       : {args.d_model}")
    print(f"  Num Heads     : {args.num_heads}")
    print(f"  Num Layers    : {args.num_layers}")
    print(f"  d_ff          : {args.d_ff}")
    print(f"  Dropout       : {args.dropout}")
    print("="*60 + "\n")


def main():
    """
    Complete training pipeline.
    """

    # ── Parse Arguments ───────────────────────────────────
    args = parse_arguments()

    # ── Device Setup ──────────────────────────────────────
    # Use GPU if available (much faster), else CPU
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    print_config(args, device)

    # ── Step 1: Prepare Data ──────────────────────────────
    print("STEP 1: Preparing Data...")
    print("-"*40)

    train_loader, val_loader, test_loader, tokenizer = prepare_data(
        questions_path = args.questions,
        answers_path   = args.answers,
        num_samples    = args.samples,
        max_length     = args.max_length,
        batch_size     = args.batch_size
    )

    # ── Step 2: Build Model ───────────────────────────────
    print("\nSTEP 2: Building Model...")
    print("-"*40)

    model = MiniGPT(
        vocab_size  = len(tokenizer),
        d_model     = args.d_model,
        num_heads   = args.num_heads,
        num_layers  = args.num_layers,
        d_ff        = args.d_ff,
        max_seq_len = args.max_length,
        dropout     = args.dropout
    ).to(device)

    # ── Step 3: Train ─────────────────────────────────────
    print("\nSTEP 3: Training...")
    print("-"*40)

    # Store model config so we can rebuild it during inference
    model_config = {
        'vocab_size'  : len(tokenizer),
        'd_model'     : args.d_model,
        'num_heads'   : args.num_heads,
        'num_layers'  : args.num_layers,
        'd_ff'        : args.d_ff,
        'max_seq_len' : args.max_length,
        'dropout'     : args.dropout,
        # Training settings
        'num_epochs'    : args.epochs,
        'learning_rate' : args.lr,
        'batch_size'    : args.batch_size,
        'patience'      : args.patience,
        'pad_token_id'  : tokenizer.pad_token_id,
        'num_samples'   : args.samples,
    }

    history = train_model(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        tokenizer    = tokenizer,
        config       = model_config,
        device       = device
    )

    # ── Step 4: Test Evaluation ───────────────────────────
    print("\nSTEP 4: Final Test Evaluation...")
    print("-"*40)

    test_loss, test_ppl, test_bleu = run_test_evaluation(
        checkpoint_path = 'weights/best_model.pt',
        questions_path  = args.questions,
        answers_path    = args.answers,
        num_samples     = args.samples
    )

    # ── Step 5: Final Summary ─────────────────────────────
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"  Best Train Loss : {min(history['train_loss']):.4f}")
    print(f"  Best Val Loss   : {min(history['val_loss']):.4f}")
    print(f"  Best Val PPL    : {min(history['val_ppl']):.2f}")
    print(f"  Best Val BLEU   : {max(history['val_bleu']):.4f}")
    print("-"*60)
    print(f"  Test Loss       : {test_loss:.4f}")
    print(f"  Test Perplexity : {test_ppl:.2f}")
    print(f"  Test BLEU       : {test_bleu:.4f}")
    print("="*60)
    print("\nGraphs saved in  → graphs/")
    print("Best model saved → weights/best_model.pt")
    print("\nTraining complete! Now run:")
    print("  python test_eval.py --mode interactive")
    print("  python test_eval.py --mode batch")


if __name__ == "__main__":
    main()
