import torch
import torch.nn as nn
from config.pytorch.text_config import Config

class Model(nn.Module):
  def __init__(self, vocab_size, embedding_size, hidden_size, n_layers, n_classes):
    super(Model, self).__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_size)
    self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True)
    self.out = nn.Linear(hidden_size, n_classes)
  
  def forward(self, text):
    embedded = self.embedding(text)
    output, (h_0, c_0) = self.rnn(embedded) #output = [B, time_stamp, hidden_size]
    h_0=h_0.squeeze(0)
    out = self.out(h_0)
    return out
  
  def opt(self, model):

    if Config.optimizer['name'] == 'AdamW':
      param_optimizer = list(model.named_parameters())
      no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

      optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
      ]
      optimizer = getattr(torch.optim, Config.optimizer['name'])(optimizer_grouped_parameters, **Config.optimizer['params'])
    
    if Config.optimizer['name'] == 'Adam':
      optimizer = getattr(torch.optim, Config.optimizer['name'])(model.named_parameters(), **Config.optimizer['params'])

    return optimizer
  
  def loss(self):
    criterion = getattr(nn, Config.loss['name'])()
    return criterion