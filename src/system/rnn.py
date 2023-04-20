import torch
import torch.nn as nn


class SystemRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SystemRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Define the weight matrices
        self.Wx = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Wh = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b = torch.nn.Parameter(torch.randn(hidden_size))
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input_signal):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        output = torch.empty(len(input_signal.T))
        h = torch.zeros(self.hidden_size).to(device)

        for index, x in enumerate(input_signal.T):
            # Compute the weighted sum of the input and hidden state
            hx = torch.matmul(x, self.Wx)
            hh = torch.matmul(h, self.Wh)

            # Compute the new hidden state
            h = self.softmax(hx + hh + self.b)

            # Compute output
            output[index] = self.fc(h)

        return output

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)