
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