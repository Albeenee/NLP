from collections import Counter
import torch
import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, DataCollatorWithPadding




def create_dataset(df, labels, hf_token):

    # Séparer les données en train/test sans sklearn
    train_df = df[:int(0.8 * len(df))]
    test_df = df[int(0.8 * len(df)):]


    # Calcul des poids inverses de fréquence
    label_counts = Counter(train_df['label'])
    total_count = sum(label_counts.values())
    class_weights = [total_count / label_counts[i] for i in range(len(labels))]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)



    # Convertir en Dataset
    train_ds = Dataset.from_pandas(train_df[['input', 'label']])
    test_ds = Dataset.from_pandas(test_df[['input', 'label']])

    # TOKENIZATION
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=hf_token)

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


    return train_ds, test_ds, data_collator, class_weights_tensor

