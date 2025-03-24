"""
ICOS Intelligence Coordination
----------------------------------------
The ICOS Coordination API has two goals:
a) First, models can be pre-built and added to the API as specified in a Developer guide. The API outputs model predictions or information about a new model trained in this scenario. This is performed for easy integration of ML models with automated functions of the OS developed in ICOS.
b) Second, part of this API is targeted to extend ML libraries to make them available to a technical user to save storage resources in devices with access to the API. In this context, the API returns a framework environment to allow users easy plug-and-play with the environment already available in the API.

Copyright © 2022-2025 CeADAR Ireland

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see <http://www.gnu.org/licenses/>.

This work has received funding from the European Union's HORIZON research 
and innovation programme under grant agreement No. 101070177.
----------------------------------------
"""
import os
import time
import numpy as np
import torch
import logging
import torch.nn as nn
from torch import optim
from dataclay import DataClayObject, activemethod
import sys
from dataset import root
from .lstm_model import *
from .utils import *
from typing import Any
from tqdm import tqdm as tqdm_progress
from dataclasses import dataclass
from memory_profiler import memory_usage

@dataclass
class ApplicationArgs:
    input_size: int = 2
    output_size: int = 2
    hidden_size: int = 64
    num_epochs: int = 100
    batch_size: int = 64
    dry_run: bool = False
    loop_back: int = 6
    name_experiment: str = "Mac"
    device: str = "cuda"

class MetricsUtilisation(DataClayObject):
    args: ApplicationArgs
    device: torch.device 

    def __init__(self, **args):
        super().__init__()
        self.args = ApplicationArgs(**args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
    
    def train_lstm_model(self, num_epochs):
        self.model.to(self.device)
        self.criterion.to(self.device)
        
        start_time = time.time()
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
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
        self.model.to(self.device)
        
        start_time = time.time()
        for epoch in range(num_epochs):
            self.model.eval()
            eval_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    eval_loss += loss.item()
            avg_eval_loss = eval_loss / len(self.val_loader)
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], Eval Loss: {avg_eval_loss:.4f}")
        eval_time = time.time() - start_time
        return avg_eval_loss, eval_time
    
    def save_csv(self, data, results_dir, filename):
        df = pd.DataFrame(data, columns=["y"])
        df.to_csv(os.path.join(results_dir, filename), index=False)

    @activemethod
    def train(self, results_dir):
        results_dir = os.path.join("results", results_dir, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(results_dir, exist_ok=True)
        print("Saving in...", results_dir)
        logging.info(f"Device: {self.device}")
        
        batch_size = self.args.batch_size
        dataset = "node_3_utilisation_sample_dataset.csv"
        DATASET_PATH = os.path.join(root, dataset)
        logging.info(f'Dataset: {DATASET_PATH}')
        
        logging.info("Collecting raw data from local device")
        raw_data = pd.read_csv(DATASET_PATH)
        raw_data_clean = raw_data.set_index('timestamp')
        train_df, test_df = data_simple_split(raw_data_clean, test_size=0.2)
        
        self.data_components = prepare_data(train_df=train_df, test_df=test_df, look_back=self.args.loop_back, batch_size=self.args.batch_size)
        
        self.train_loader, self.val_loader = create_dataloaders(self.data_components['train_dataset'],
                                                                self.data_components['test_dataset'],
                                                                batch_size=batch_size)
        self.model = LSTMModel(self.args.input_size, self.args.hidden_size, self.args.output_size).to(self.device)
        
        logging.info(self.model)
        print_size_of_model(self.model, "fp32")
        
        self.criterion = nn.MSELoss().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        mem_usage = memory_usage((self.train_lstm_model, (self.args.num_epochs,)), interval=0.1)
        peak_memory_usage = sum(mem_usage) / len(mem_usage)

        train_loss, training_time = self.train_lstm_model(self.args.num_epochs)
        eval_loss, eval_time = self.eval_lstm_model(self.args.num_epochs)
        
        outputs = self.model(self.data_components['X_test'].to(self.device))
        y_pred = outputs.cpu().detach().numpy().reshape(outputs.shape[0], outputs.shape[1])
        y_pred = self.data_components['scaler_obj'].inverse_transform(y_pred)
        y_test = self.data_components['scaler_obj'].inverse_transform(self.data_components['y_test'].squeeze())

        y_pred_cpu = y_pred[:, 0].reshape(-1, 1) 
        y_pred_mem = y_pred[:, 1].reshape(-1, 1)
        y_test_cpu = y_test[:, 0].reshape(-1, 1)
        y_test_mem = y_test[:, 1].reshape(-1, 1)
        
        metrics_cpu = metrics_pytorch(self.model, y_test_cpu, y_pred_cpu)
        metrics_mem = metrics_pytorch(self.model, y_test_mem, y_pred_mem)
        
        metrics_cpu['Memory Usage (MB)'] = peak_memory_usage
        metrics_mem['Memory Usage (MB)'] = peak_memory_usage
        
        logging.info(f"Metrics CPU {metrics_cpu}")
        logging.info(f"Metrics MEM {metrics_mem}")
        print("Metrics CPU: ", metrics_cpu)
        print("Metrics MEM: ", metrics_mem)            
        
        model_path = os.path.join(results_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)
        
        metrics_cpu['metric'] = 'cpu'
        metrics_mem['metric'] = 'mem'
        metrics_df = pd.DataFrame([metrics_cpu, metrics_mem])

        metrics_df['training_time'] = training_time
        metrics_df['eval_time'] = eval_time

        cols = ['metric'] + [col for col in metrics_df.columns if col != 'metric']
        metrics_df = metrics_df[cols]




        metrics_df.to_csv(os.path.join(results_dir, "metrics_combined.csv"), index=False)
        
        self.save_csv(y_pred_cpu, results_dir, "y_pred_cpu.csv")
        self.save_csv(y_pred_mem, results_dir, "y_pred_mem.csv")
        self.save_csv(y_test_cpu, results_dir, "y_test_cpu.csv")
        self.save_csv(y_test_mem, results_dir, "y_test_mem.csv")

        return metrics_cpu, metrics_mem, train_loss, eval_loss, training_time, eval_time

    @activemethod
    def predict(self, sample_data):
        logging.info("Run prediction on trained model")
        logging.info(f"Predicting 1-step future value")
        data_dict = {"CPU": sample_data['input_1'],
                     "MEM": sample_data['input_2']}
        data = pd.DataFrame(data=data_dict)
        input_data_scaled = self.data_components['scaler_obj'].transform(data.values)
        input_data_scaled = np.expand_dims(np.array(input_data_scaled.reshape(1,-1)),2)
        input_data_scaled = torch.from_numpy(input_data_scaled).float()
        start_time = time.time()
        output_data_scaled = self.model(input_data_scaled)
        inferencing_time = time.time() - start_time
        output_data_scaled = output_data_scaled.detach().cpu().numpy()
        prediction = self.data_components['scaler_obj'].inverse_transform(output_data_scaled.reshape(1,-1))
        logging.info(f"Predictions: {prediction}")
        logging.info(f"Inferencing Time: {inferencing_time:.4f} seconds")
        return prediction, inferencing_time
