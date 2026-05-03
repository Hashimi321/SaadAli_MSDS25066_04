
# explore_data.py
# PURPOSE: Understand our dataset before building anything

import pandas as pd

print("Loading CSV files...")
print("="*60)

questions = pd.read_csv('dataset/Questions.csv', encoding='latin1')
answers   = pd.read_csv('dataset/Answers.csv',   encoding='latin1')
tags      = pd.read_csv('dataset/Tags.csv',       encoding='latin1')

print("\n--- SHAPES ---")
print(f"Questions : {questions.shape}")
print(f"Answers   : {answers.shape}")
print(f"Tags      : {tags.shape}")

print("\n--- COLUMNS ---")
print(f"Questions : {questions.columns.tolist()}")
print(f"Answers   : {questions.columns.tolist()}")
print(f"Tags      : {tags.columns.tolist()}")

print("\n--- MISSING VALUES IN QUESTIONS ---")
print(questions.isnull().sum())

print("\n--- MISSING VALUES IN ANSWERS ---")
print(answers.isnull().sum())

print("\n--- SAMPLE QUESTION ROW ---")
for col in questions.columns:
    val = str(questions[col].iloc[0])
    print(f"  {col}: {val[:200]}")

print("\n--- SAMPLE ANSWER ROW ---")
for col in answers.columns:
    val = str(answers[col].iloc[0])
    print(f"  {col}: {val[:200]}")

print("\n--- HOW ARE THEY LINKED? ---")
if 'AcceptedAnswerId' in questions.columns:
    has_accepted = questions['AcceptedAnswerId'].notna().sum()
    print(f"Questions WITH accepted answer  : {has_accepted}")
    print(f"Questions WITHOUT accepted answer: {len(questions) - has_accepted}")
else:
    print("No AcceptedAnswerId column!")
    print("Columns available:", questions.columns.tolist())

print("\n--- SAMPLE TAG ROW ---")
for col in tags.columns:
    val = str(tags[col].iloc[0])
    print(f"  {col}: {val[:200]}")

print("\n" + "="*60)
print("DONE! Copy this output and share with mentor.")