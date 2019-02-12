import torch.nn as nn
import torch


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(DQN, self).__init__()
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = self.input_to_hidden(x)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.hidden_to_output(out)
        out = self.dropout(out)
        return out
