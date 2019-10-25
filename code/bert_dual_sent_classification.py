import torch.nn as nn
import torch

from transformers import BertPreTrainedModel
from transformers import BertModel
from torch.nn.parameter import Parameter
from torch.nn import CrossEntropyLoss


class BertForDualSent(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForDualSent, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pre_linear = nn.Linear(config.hidden_size, 300)
        self.activation = nn.SELU()
        self.reduce_fuse_linear = nn.Linear(600, 300)
        self.cos = nn.CosineSimilarity()
        
        self.rank_margin = nn.MarginRankingLoss(margin=0.4)

        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.init_weights()




    def forward(self, sent1, sent2, labels=None):
        _, sent1 = self.bert(sent1)
        sent1 = self.activation(sent1)
        sent1 = self.pre_linear(sent1)
        sent1 = self.dropout(sent1)

        # try:
        ori = sent2
        _, sent2 = self.bert(sent2)
        sent2 = self.activation(sent2)
        sent2 = self.pre_linear(sent2)
        sent2 = self.dropout(sent2)

        fused_sent = torch.cat((sent1+sent2, sent1*sent2),dim=1)
        fused_sent = self.reduce_fuse_linear(fused_sent)

        cos_simi1 = self.cos(fused_sent, sent1)
        cos_simi2 = self.cos(fused_sent, sent2)

        outputs = (cos_simi1, cos_simi2), None, None
        if labels is not None:
            labels[labels==0] = -1
            # sent1 - sent2 > 0.4
            loss_rank = self.rank_margin(cos_simi1, cos_simi2, labels)

            outputs = (loss_rank,) + outputs
        return outputs