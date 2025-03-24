import logging
import sys
import os
import time 
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../AIoffload')))
from model.metrics_utilisation import MetricsUtilisation

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Train the AI model using MetricsUtilisation.')
    parser.add_argument('--input_size', type=int, default=2, help='Input size of the model')
    parser.add_argument('--output_size', type=int, default=2, help='Output size of the model')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size of the model')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--dry_run', action='store_true', help='Run a dry run without actual training')
    parser.add_argument('--name_experiment', type=str, default='Mac', help='Name of the experiment')

    args = parser.parse_args()
    results_dir = args.name_experiment  # Save results in the same directory as the script
    # base_dir = os.path.dirname(os.path.abspath/(__file__))
    # results_dir = os.path.join(base_dir, args.name_experiment)
    # os.makedirs(results_dir, exist_ok=True)

    args_dict = vars(args)
    mu = MetricsUtilisation(**args_dict)

    try:
        metrics_cpu, metrics_mem, train_loss, eval_loss, training_time, eval_time = mu.train(results_dir)
        print(f"metrics_cpu: {metrics_cpu}\n metrics_mem: {metrics_mem}")
        print(f"Training Time: {training_time:.4f}, Eval Time: {eval_time:.4f}")
        print("[client] Memory consumed during process:", metrics_cpu['Memory Usage (MB)'])
    except Exception as e:
        print("Error during training:", e)

if __name__ == "__main__":
    main()