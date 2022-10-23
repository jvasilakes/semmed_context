import torch
import torch.nn as nn
from transformers import AutoModel


class BertPredicationFilter(nn.Module):

    def __init__(self, bert_model_name_or_path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",  # noqa
                 dropout_prob=0.1):
        super().__init__()
        self.bert_model_name_or_path = bert_model_name_or_path
        self.bert = AutoModel.from_pretrained(self.bert_model_name_or_path)
        self.drop = nn.Dropout(dropout_prob, False)
        self.fc_out = nn.Linear(768, 2)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, **model_input):
        outputs = self.bert(**model_input)
        pooled_output = outputs.pooler_output
        pooled_output = self.drop(pooled_output)
        logits = self.fc_out(pooled_output)
        return logits

    def predict_from_logits(self, logits):
        """
        logits is just the output of self.forward()
        """
        with torch.no_grad():
            preds = logits.argmax(dim=1)
        return preds
