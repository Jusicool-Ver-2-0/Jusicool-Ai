from torch import nn
import torch
class GRU(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(GRU, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.63)
        self.fc_1 = nn.Linear(hidden_size, 256)
        self.fc = nn.Linear(256, 128)
        self.fc_2 = nn.Linear(128, 1)
        self.BTNorm = nn.BatchNorm1d(num_features=256)
        self.BTNorm1 = nn.BatchNorm1d(num_features=128)
        self.BTNorm2 = nn.BatchNorm1d(num_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.63)
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        output, hn = self.gru(x, h_0)
        hn = hn[-1]
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.BTNorm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.BTNorm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_2(out)
        out = self.sigmoid(out)

        return out