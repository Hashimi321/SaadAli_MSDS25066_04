
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