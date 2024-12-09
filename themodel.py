import torch
import torch.nn as nn
import torch.optim as optim

class localLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, lr, dropout_prob):
        super(localLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.lr = lr

    def forward(self, x):
        lstm_out, _ = self.lstm(x) 
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out) 
        return out
    def loss(self, y_hat, y):
      fn = nn.BCEWithLogitsLoss()
      y = y.unsqueeze(1).float()
      return fn(y_hat, y)
    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), self.lr)