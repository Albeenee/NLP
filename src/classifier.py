from typing import List

import torch
import pandas as pd
from datasets import Dataset
from transformers import RobertaTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
import numpy as np
import os

from transformers import RobertaConfig
from create_datasets import create_dataset
from model import RobertaForWeightedClassification
from compute_metrics import compute_metrics


class Classifier:
    """
    The Classifier: complete the definition of this class template by completing the __init__() function and
    the 2 methods train() and predict() below. Please do not change the signature of these methods
     """

    ############################################# complete the classifier class below
    
    def __init__(self, ollama_url: str):
        """
        This should create and initialize the model.
        !!!!! If the approach you have choosen is in-context-learning with an LLM from Ollama, you should initialize
         the ollama client here using the 'ollama_url' that is provided (please do not use your own ollama
         URL!)
        !!!!! If you have choosen an approach based on training an MLM or a generative LM, then your model should
        be defined and initialized here.
        """

        HF_TOKEN = 'XXXX'
        os.environ["WANDB_MODE"] = "disabled"

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", token=HF_TOKEN)
        self.model = None  # Initialized during train()
        self.trainer = None
        self.label2id = {}
        self.id2label = {}

        os.environ["WANDB_MODE"] = "disabled"
        self.device = torch.device("cpu") # default
        

    
    
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
        df = pd.read_csv(train_filename, delimiter='\t', on_bad_lines='skip',
                         header=None, names=['label', 'catégorie', 'heure', 'origin', 'texte'])

        self.device = device
        train_ds, test_ds, class_weights_tensor, labels = create_dataset(df)

        self.label2id = {label: i for i, label in enumerate(labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}

        config = RobertaConfig.from_pretrained("roberta-base", num_labels=len(labels))
        self.model = RobertaForWeightedClassification.from_pretrained(
            "roberta-base",
            config=config,
            class_weights=class_weights_tensor
        ).to(device)

        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_strategy="no",
            report_to=None,
            remove_unused_columns=False
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
        )

        self.trainer.train()


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
        If the approach you have choosen is in-context-learning with an LLM from Ollama, ignore the 'device'
        parameter (because the device is specified when launching the Ollama server, and not by the client side)
        Otherwise:
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

        df = pd.read_csv(data_filename, delimiter='\t', on_bad_lines='skip',
                header=None, names=['label', 'catégorie', 'heure', 'origin', 'texte'])
        
        # Fusion des colonnes pour l'entrée texte
        df = pd.read_csv(data_filename, delimiter='\t', on_bad_lines='skip',
                         header=None, names=['label', 'catégorie', 'heure', 'origin', 'texte'])

        df['input'] = df['catégorie'].astype(str) + ' : ' + df['texte'].astype(str)
        df = df.dropna(subset=['input'])

        pred_ds = Dataset.from_pandas(df[['input']])

        def tokenize_function(example):
            return self.tokenizer(example["input"], padding="max_length", truncation=True)

        pred_ds = pred_ds.map(tokenize_function, batched=False)
        pred_ds = pred_ds.remove_columns([col for col in pred_ds.column_names if col not in ['input_ids', 'attention_mask']])
        pred_ds.set_format("torch")

        self.model.to(device)

        predictions = self.trainer.predict(pred_ds)
        pred_labels = np.argmax(predictions.predictions, axis=-1)

        return [self.id2label[i] for i in pred_labels]






