
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