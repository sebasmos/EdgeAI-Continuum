import logging
import sys
import os
import os
import time
import numpy as np
import torch
import logging
import torch.nn as nn
from dataclay import DataClayObject, activemethod
import sys
from dataclasses import dataclass
import pandas as pd

@dataclass
class ApplicationArgs:
    input_size: int = 1
    output_size: int = 1
    hidden_size: int = 64
    num_epochs: int = 100
    batch_size: int = 64
    dry_run: bool = False
    loop_back: int = 6

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output
    
class MetricsUtilisation(DataClayObject):
    args: ApplicationArgs

    def __init__(self, **args):
        self.args = ApplicationArgs(**args)
        
    def train_lstm_model(self, num_epochs):
        start_time = time.time()
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(self.train_loader)
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")
        training_time = time.time() - start_time
        return avg_train_loss, training_time
    
    def eval_lstm_model(self, num_epochs):
        start_time = time.time()
        for epoch in range(num_epochs):
            self.model.eval()
            eval_loss = 0
            for X_batch, y_batch in self.val_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                eval_loss += loss.item()
            avg_eval_loss = eval_loss / len(self.val_loader)
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Eval Loss: {avg_eval_loss:.4f}")
        eval_time = time.time() - start_time
        return avg_eval_loss, eval_time
    
    def save_csv(self, data,results_dir, filename):
        df = pd.DataFrame(data, columns=["y"])
        df.to_csv(os.path.join(results_dir, filename), index=False)

    @activemethod
    def train(self):

        self.model = LSTMModel(self.args.input_size, self.args.hidden_size, self.args.output_size)
        print("checking the model...")
        print(self.args.input_size, )
        logging.info(self.model)
        return 0, 0, 0, 0, 0, 0

def main():
    logging.basicConfig(level=logging.INFO)
    
    args = {
        'input_size': 2,
        'output_size': 2,
        'hidden_size': 64,
        'num_epochs': 100,
        'batch_size': 64,
        'dry_run': False
    }
    
    mu = MetricsUtilisation(**args)
     
    try:
        metrics_cpu, metrics_mem, train_loss, eval_loss, training_time, eval_time = mu.train()

    except Exception as e:
        print("Error during training:", e)
    

if __name__ == "__main__":
    main()