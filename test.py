
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