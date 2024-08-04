#!/usr/bin/env bash

current_path=$(pwd)

# Find the path containing PFLlib, excluding hidden directories
pfllib_parent=$(find "$current_path" -type d -name "PFLlib" -not -path '*/\.*' | head -n 1 | xargs dirname)

if [ -z "$pfllib_parent" ]; then
    echo "Error: PFLlib directory not found in $HOME"
    exit 1
fi

change_path="$pfllib_parent/PFLlib/system"

if [ "$current_path" != "$change_path" ]; then
    if [ -d "$change_path" ]; then
        cd "$change_path"
        echo "Changed to: $(pwd)"
    else
        echo "Error: $change_path does not exist"
        exit 1
    fi
else
    echo "Already in the correct directory"
fi


python main.py --dataset $1 \
               --model cnn \
               --algorithm FedAvg \
               --global_rounds 200 \
               --num_clients 50 \
               --device_id 0 \
               --num_classes 10 \
               --device cuda \
               --goal train \
               --batch_size 20 \
               --local_epochs 10 \
               --local_learning_rate 0.05 \
               --join_ratio $2 \
               --learning_rate_decay True \
               --learning_rate_decay_gamma 0.99 \
               --client_drop_rate $3 


echo "Change back to the previous directory: $current_path"
cd $current_path