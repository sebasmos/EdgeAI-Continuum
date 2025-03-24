#!/bin/bash

# Run the command 20 times
for i in {1..20}
do
    python train_dataclay.py --name_experiment "Mac-OrangePI_server" --client_experiment "./Mac-OrangePI_client" --server_ip "137.43.50.183"
done