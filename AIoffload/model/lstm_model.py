import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModelDeep(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.3):
        super(LSTMModelDeep, self).__init__()
        
        self.hidden_layers = hidden_layers
        self.lstm_layers = nn.ModuleList()
        
        # First LSTM layer
        self.lstm_layers.append(nn.LSTM(input_size, hidden_layers[0], batch_first=True))
        
        # Additional LSTM layers
        for i in range(1, len(hidden_layers)):
            self.lstm_layers.append(nn.LSTM(hidden_layers[i-1], hidden_layers[i], batch_first=True))
        
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for hidden_size in hidden_layers])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in hidden_layers])
        
        self.fc = nn.Linear(hidden_layers[-1], output_size)
    
    def forward(self, x):
        for i, lstm in enumerate(self.lstm_layers):
            x, _ = lstm(x)
            x = self.batch_norms[i](x[:, -1, :])
            x = self.dropouts[i](x)
        
        output = self.fc(x)
        return output


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        # output = self.fc(output[:,-2:,:])
        output = self.fc(output[:, -1, :])
        return output
    
class LighterStudentLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LighterStudentLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.gru(x)
        
        output = self.fc(output[:,-2:,:]) # 1585,2,1
        # output = self.fc(output[:, -1, :])
        
        return output


if __name__ == "__main__":
    input_size = 1
    hidden_layers = [200, 100, 50] 
    output_size = 1
    seq_length = 3  
    batch_size = 64 

    # model = LSTMModel(input_size, 50, output_size)
    model = LSTMModelDeep(input_size, hidden_layers, output_size)
    
    dummy_input = torch.randn(batch_size, seq_length, input_size)

    output = model(dummy_input)

    # Print the output shape
    print("Output shape:", output.shape)  # Expected: (batch_size, output