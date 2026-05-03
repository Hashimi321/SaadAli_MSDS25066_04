
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