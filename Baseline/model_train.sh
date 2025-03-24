#!/bin/bash

# Run the command 20 times
for i in {1..20}
do
    python model_train.py --name_experiment "OrangePI_with_Mac"
done