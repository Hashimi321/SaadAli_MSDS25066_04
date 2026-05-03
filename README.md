
# Assignment 4 - Domain-Specific Transformer for QA
**Author:** Saad Ali  
**Roll Number:** MSDS25066  
**Course:** Deep Learning - Spring 2026

## Project Structure
SaadAli_MSDS25066_04/
├── msds25066_04_task1.py   # Main pipeline
├── model.py                # MiniGPT architecture
├── data_utils.py           # Data loading & preprocessing
├── train.py                # Training loop
├── test.py                 # Test evaluation
├── test_eval.py            # Interactive QA chatbot
├── MSDS25066_04_allCode.py # All code combined
├── dataset/                # CSV files (not submitted)
├── weights/                # Saved model checkpoints
├── graphs/                 # Training curves & outputs
├── requirements.txt        # Dependencies
└── README.md               # This file

## Setup Instructions

### 1. Create virtual environment
```bash
python -m venv dl_env
dl_env\Scripts\activate        # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## How to Run

### Train the model
```bash
# Default settings (recommended for CPU)
python msds25066_04_task1.py

# Custom settings
python msds25066_04_task1.py --samples 3000 --epochs 15 --d_model 128

# With custom dataset path
python msds25066_04_task1.py --questions path/to/Questions.csv --answers path/to/Answers.csv
```

### Run interactive QA chatbot
```bash
python test_eval.py --mode interactive
```

### Run batch generation examples
```bash
python test_eval.py --mode batch
```

### Run test evaluation only
```bash
python test.py
```

## Model Architecture
- Decoder-only Transformer (GPT-style)
- Embedding + Positional Encoding
- Masked Multi-Head Self-Attention
- Feed Forward Network with residuals
- CrossEntropyLoss + AdamW + CosineAnnealingLR

## Dataset
Stack Overflow Python Q&A subset
- Questions.csv, Answers.csv, Tags.csv
- encoding = latin1
- Link questions to best answers via ParentId + Score