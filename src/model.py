from transformers import RobertaForSequenceClassification
from torch.nn import CrossEntropyLoss

class RobertaForWeightedClassification(RobertaForSequenceClassification):
    def __init__(self, config, class_weights=None):
        super().__init__(config)
        if class_weights is not None:
            self.class_weights = class_weights
            self.loss_fct = CrossEntropyLoss(weight=self.class_weights)
        else:
            self.loss_fct = CrossEntropyLoss()

    def forward(self, **kwargs):
        # Supprimer les arguments inattendus
        if "num_items_in_batch" in kwargs:
            kwargs.pop("num_items_in_batch")

        outputs = super().forward(**kwargs)

        if "labels" in kwargs:
            loss = self.loss_fct(outputs.logits, kwargs["labels"])
            return (loss, outputs.logits)
        return outputs