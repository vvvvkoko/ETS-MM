from transformers import AutoModel
import torch.nn as nn
from torch_geometric.nn.models import MLP
import torch


class LM_Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.LM_model_name = model_config['lm_model']
        if self.LM_model_name == 'deberta':
            self.LM = AutoModel.from_pretrained('...')  # setting deberta path
        elif self.LM_model_name == 'roberta':
            self.LM = AutoModel.from_pretrained('...')   # setting roberta path
        elif self.LM_model_name == 'bert':
            self.LM = AutoModel.from_pretrained('...')  # setting bert path
        elif self.LM_model_name == 'roberta-f':
            self.LM = AutoModel.from_pretrained('...')  # setting roberta-f path
        else:
            raise ValueError()

        self.classifier = MLP(in_channels=self.LM.config.hidden_size,
                              hidden_channels=model_config['classifier_hidden_dim'], out_channels=2,
                              num_layers=model_config['classifier_n_layers'], act=model_config['activation'])

        self.LM.config.hidden_dropout_prob = model_config['lm_dropout']
        self.LM.attention_probs_dropout_prob = model_config['att_dropout']

    def forward(self, tokenized_tensors):
        out = self.LM(output_hidden_states=True, **tokenized_tensors)['hidden_states']
        embedding = out[-1].mean(dim=1)

        return embedding.detach(), self.classifier(embedding)