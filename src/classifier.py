from typing import List

import torch
from torch.utils.data import DataLoader
import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, TrainingArguments, Trainer


import numpy as np
import os

from transformers import RobertaConfig

from create_dataset import create_dataset, insert_word_tags
from model import RobertaForTaggedWordClassification
from compute_metrics import compute_metrics
hf_token = os.getenv("HF_TOKEN")


class Classifier:
    """
    The Classifier: complete the definition of this class template by completing the __init__() function and
    the 2 methods train() and predict() below. Please do not change the signature of these methods
     """

    ############################################# complete the classifier class below
    
    def __init__(self, ollama_url: str):
        """
        This should create and initialize the model.
        This should create and initialize the model.
        !!!!! If the approach you have choosen is in-context-learning with an LLM from Ollama, you should initialize
         the ollama client here using the 'ollama_url' that is provided (please do not use your own ollama
         URL!)
        !!!!! If you have choosen an approach based on training an MLM or a generative LM, then your model should
        be defined and initialized here.
        """

        os.environ["WANDB_MODE"] = "disabled"

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=hf_token)
        self.tokenizer.add_tokens(["<W>", "</W>"])
        self.model = None  # Initialized during train()
        self.trainer = None
        self.label2id = {}
        self.id2label = {}
      

    
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
        If the approach you have choosen is in-context-learning with an LLM from Ollama, you must
          not train the model, and this method should contain only the "pass" instruction
        Otherwise:
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS

        """
        # Load the dataset
        df = pd.read_csv(train_filename, delimiter='\t', on_bad_lines='skip',
                         header=None, names=['label', 'catégorie', 'word', 'heure', 'texte'])

        # Create datasets
        train_ds, test_ds, data_collator, class_weights_tensor, n_labels, label2id, id2label = create_dataset(df, self.tokenizer)
        self.label2id = label2id
        self.id2label = id2label

        # Model and tokenizer
        config = RobertaConfig.from_pretrained("roberta-large", num_labels=n_labels)
        

        self.model = RobertaForTaggedWordClassification.from_pretrained(
            "roberta-large",
            config=config,
            tokenizer=self.tokenizer,
            class_weights=class_weights_tensor
        )

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="no",
            report_to=None,  # Desactivate WandB
            remove_unused_columns=False  # Desactivate warning
        )


        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics,
            data_collator=data_collator,  # remplace l'ancien tokenizer
        )

        self.trainer.train()


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
        If the approach you have choosen is in-context-learning with an LLM from Ollama, ignore the '  '
        parameter (because the device is specified when launching the Ollama server, and not by the client side)
        Otherwise:
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

        # Load and preprocess the data
        df = pd.read_csv(data_filename, delimiter='\t', on_bad_lines='skip',
                        header=None, names=['label', 'catégorie', 'word', 'heure', 'texte'])

        df["texte"] = df.apply(insert_word_tags, axis=1)
        df['input'] = df['catégorie'].astype(str) + ' : ' + df['texte'].astype(str)
        df = df.dropna(subset=['input', 'label'])

        # Prepare dataset for prediction
        pred_ds = Dataset.from_pandas(df[['input']])

        # Tokenize using same logic as in training
        def tokenize(example):
            return self.tokenizer(
                example["input"],
                padding="max_length",
                truncation=True,
                return_attention_mask=True
            )

        pred_ds = pred_ds.map(tokenize, batched=True)
        pred_ds = pred_ds.remove_columns([col for col in pred_ds.column_names if col not in ['input_ids', 'attention_mask']])
        pred_ds.set_format("torch")

        # Set model to eval and move to device
        self.model.to(device)
        self.model.eval()

        # Use trainer to predict
        predictions = self.trainer.predict(pred_ds, metric_key_prefix="test", ignore_keys=["labels"])
 
        # Get the predicted label indices
        pred_labels = np.argmax(predictions.predictions, axis=-1)

        # Convert IDs back to labels
        return [self.id2label[i] for i in pred_labels]
