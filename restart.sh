#!/bin/bash
while true; do
  torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=29501 train_multigpu.py
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
    break
  else
    echo "Training crashed with exit code $EXIT_CODE. Restarting..."
    sleep 10  # Optional delay before restarting
  fi
done
