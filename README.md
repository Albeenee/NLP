# NLP

## Contributors
- Theau d'Audiffret
- Tanguy Le Cloirec
- Albane Vigier  

---

## Model Overview

This project implements a custom classifier based on the `roberta-base` model from Hugging Face's Transformers library. The dataset contains 4 features: 
- the aspect category on which the opinion is expressed, 
- a specific target term, 
- the character offsets of the term (start:end)
- and the sentence in which the term occurs and the opinion is expressed.

The task is to classify the input into classes of polarity: positive, negative or neutral.

### Classifier Type
- **Model**: Fine-tuned Transformer (`roberta-base`)
- **Architecture**: `RobertaForTaggedWordClassification` (custom model class built on top of `RobertaModel`)
- **Classification Head**: Custom linear layer over the hidden state of the word tagged with `<W>` and `</W>` tokens. Indeed, by adding this new representation with tokens, we can just feed the classifier with the contextual embedding of the token of interest (target term) instead of the entire embedding of the sentence. This way the classifier stresses on the target term. If there are multiple embedded tokens between the embedded tags, we average pool to have only one embedding.

### Input Representation
- The input format is:  [category] : [text with <W>word</W> tags]
- The tokenizer is extended to include two special tokens: `<W>` and `</W>` that mark the target word in the sentence.

### Label Encoding
- Labels are mapped to numeric indices using a `label2id` dictionary.
- Inverse mapping (`id2label`) is used to convert predicted indices back to label names.

### Class Imbalance Handling
- Class weights are computed and passed to the loss function to account for imbalanced label distribution in the training data.

---

## Dev Set Accuracy

After running the classifier for multiple runs, we obtained the following results:
RUN 1: 86.44
RUN 2: 85.64
RUN 3: 85.64
RUN 4: 85.64
RUN 5: 85.64

Mean Dev Acc.: 85.80 (0.32)
