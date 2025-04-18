from collections import Counter
import torch
import pandas as pd
from datasets import Dataset
from transformers import DataCollatorWithPadding



def insert_word_tags(row):
    word = row["word"]
    text = row["texte"]
    # Insertion des balises <W> et </W> autour du mot
    if word in text:
        return text.replace(word, f"<W>{word}</W>", 1)
    else:
        return "ERROR"  # fallback to "ERROR" if the word is not found


def create_dataset(df, tokenizer):

    # Join the word and text columns with <W> and </W> tags
    df['input'] = df['catégorie'].astype(str) + ' : ' + df['texte'].astype(str)

    # Suppress rows with NaN values in 'input' or 'label'
    df = df.dropna(subset=['input', 'label'])

    # Encoding labels
    labels = df['label'].unique()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    df['label'] = df['label'].map(label2id)

    # Split the dataset into train and test sets
    train_df = df[:int(0.8 * len(df))]
    test_df = df[int(0.8 * len(df)):]

    # Compute class weights
    label_counts = Counter(train_df['label'])
    total_count = sum(label_counts.values())
    class_weights = [total_count / label_counts[i] for i in range(len(labels))]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)



    # Convert into dataset
    train_ds = Dataset.from_pandas(train_df[['input', 'label', 'word']])
    test_ds = Dataset.from_pandas(test_df[['input', 'label', 'word']])


    # Tokenization
    def tokenize(example):
        tokens = tokenizer(
            example["input"],
            truncation=True,
            padding="max_length",
            return_attention_mask=True
        )
        tokens["label"] = example["label"]
        return tokens

    train_ds = train_ds.map(tokenize, batched=False)
    test_ds = test_ds.map(tokenize, batched=False)


    # Suppress input column
    train_ds = train_ds.remove_columns(["input"])
    test_ds = test_ds.remove_columns(["input"])

    # Rename the column to match the Trainer and data collator expectations
    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    # Convert to torch format
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    return train_ds, test_ds, data_collator, class_weights_tensor, len(labels), label2id, id2label

