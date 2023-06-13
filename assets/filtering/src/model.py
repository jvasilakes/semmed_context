import torch.nn as nn
from transformers import AutoModel, modeling_outputs


class PubMedBERT(nn.Module):
    """
    The PubMed implementation does not include a classification head
    so one must be added. The classification head is added in the same manner
    as the other HuggingFace BERT models for consistency.
    """
    def __init__(self):
        super().__init__()
        self.weight_path = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"  # noqa
        self.bert = AutoModel.from_pretrained(self.weight_path)
        self.drop = nn.Dropout(0.1, False)
        self.fc_out = nn.Linear(768, 2, True)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.drop(pooled_output)
        logits = self.fc_out(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return modeling_outputs.SequenceClassifierOutput(
            loss=loss, logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)
