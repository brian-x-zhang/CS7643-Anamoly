import torch
import torch.nn as nn

class CLSTM(nn.Module):
    def __init__(self, conv1_out_channels=64, kernel_size=5, lstm_hidden_size=64, fc1_out_features=32):
        super(CLSTM, self).__init__()
        self.conv1 = nn.Conv1d(1, conv1_out_channels, kernel_size=kernel_size, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.conv2 = nn.Conv1d(conv1_out_channels, conv1_out_channels, kernel_size=kernel_size, stride=1, padding=2)
        self.lstm = nn.LSTM(input_size=960, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_size, fc1_out_features)
        self.fc2 = nn.Linear(fc1_out_features, 2)
        
    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1, 960)

        x, _ = self.lstm(x)
        x = self.flatten(x)
        #x = x.contiguous().view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x
