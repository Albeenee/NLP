from transformers import RobertaTokenizer, RobertaForSequenceClassification 
import torch
from torch.nn import CrossEntropyLoss
from types import SimpleNamespace


class RobertaForTaggedWordClassification(RobertaForSequenceClassification):
    def __init__(self, config, tokenizer, class_weights=None):
        super().__init__(config)
        self.roberta = self.roberta  # le backbone RoBERTa
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)
        self.tokenizer = tokenizer

        if class_weights is not None:
            self.loss_fct = CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fct = CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):

        tokenizer = self.tokenizer

        kwargs.pop("num_items_in_batch", None)
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Trouver les positions des tokens <W> et </W>
        w_token_id = tokenizer.convert_tokens_to_ids("<W>")
        end_w_token_id = tokenizer.convert_tokens_to_ids("</W>")

        # Pour chaque exemple du batch, on extrait l'embedding entre <W> et </W>
        batch_embeddings = []

        for i in range(input_ids.size(0)):
            input_seq = input_ids[i]
            hidden_seq = last_hidden_state[i]

            try:
                start_idx = (input_seq == w_token_id).nonzero(as_tuple=True)[0].item() + 1
                end_idx = (input_seq == end_w_token_id).nonzero(as_tuple=True)[0].item()

                word_embeds = hidden_seq[start_idx:end_idx]  # (mot_length, hidden)
                pooled = word_embeds.mean(dim=0)  # moyenne sur les tokens du mot
            except Exception:
                pooled = hidden_seq[0]  # fallback si les balises ne sont pas trouvées

            batch_embeddings.append(pooled)

        pooled_output = torch.stack(batch_embeddings)  # (batch_size, hidden_size)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss = self.loss_fct(logits, labels)
            return {"loss": loss, "logits": logits}
        
        return {'logits' : logits}
