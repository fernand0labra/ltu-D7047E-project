import torch
import torch.nn as nn
from var import DEVICE

class SystemRNN(nn.Module):

    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1, nonlinearity='tanh', dropout=0.0, rnn_type='vanilla'):
        super(SystemRNN, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True, nonlinearity=nonlinearity, dropout=dropout) \
            if rnn_type == 'vanilla' else \
                   nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        x, h = self.rnn(x) if h is None else self.rnn(x, h)
        return self.fc(x), h
    
    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE) \
        if self.rnn_type =='vanilla' \
        else  (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE),
               torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE))