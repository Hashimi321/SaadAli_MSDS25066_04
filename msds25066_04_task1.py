
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