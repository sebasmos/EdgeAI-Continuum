from dataclay import StubDataClayObject
from dataclay import Client
from model.utils import *
import datetime
import time
import os
import argparse

if __name__ == "__main__":
    start_time = time.time()  # Start timer
    client = None

    parser = argparse.ArgumentParser(description='Train the AI model using MetricsUtilisation.')
    parser.add_argument('--name_experiment', type=str, default='dC_Server', help='Define here the folder name for the server ')
    parser.add_argument('--client_experiment', type=str, default='dC_Client', help='Define here the folder name for the client' )
    parser.add_argument('--server_ip', type=str, default='137.43.50.183', help='Define here your SERVER IP' )

    args = parser.parse_args()

    # Experiment name will record the server-side metrics 
    name_experiment  = args.name_experiment
    server_ip  = args.server_ip

    try:
        # client = Client(proxy_host=server_ip, dataset="testdata")
        client = Client(proxy_host="137.43.203.205", dataset="testdata")
        client.start()
        
        MetricsUtilStub = StubDataClayObject["model.metrics_utilisation.MetricsUtilisation", ["args"], ["train", "predict"]]
        persistent_mt = MetricsUtilStub.get_by_alias("main_metrics")
        metrics_cpu, metrics_mem, train_loss, eval_loss, training_time, eval_time = persistent_mt.train(name_experiment)
        
        print(f"[Server] metrics_cpu: {metrics_cpu}")
        print(f"[Server] metrics_mem: {metrics_mem}")
        print(f"[Server] training_time: {training_time} s")
        print(f"[Server] eval_time: {eval_time} s")
        
    finally:
        total_proc_time = time.time() - start_time  # Compute total processing time
        m = get_memory_usage()
        print(f"[Local] Memory usage: {m} MB")
        print(f"[Local] Total processing time: {total_proc_time:.4f} s")

        # Save the Client-side memory and runtime at the client side. 
        results_dir = os.path.join("results",args.client_experiment)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"{results_dir}/{timestamp}.csv"
        
        with open(filename, "w") as f:
            f.write("memory,total proc. (s)\n")
            f.write(f"{m},{total_proc_time:.4f}\n")  # Save memory and total processing time
        
        if client is not None: # avoid dataclay shutdown error for not closing on time 
            client.stop()