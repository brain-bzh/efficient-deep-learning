#!/bin/bash

# Boucles pour tester différentes valeurs d'hyperparamètres
for batch_size in 32 64 128; do
  for epochs in 20 30; do
    for lr in 0.001 0.0005; do
      for scheduler in "" "--use_scheduler"; do
        echo "Running with batch_size=$batch_size, epochs=$epochs, lr=$lr, scheduler=$scheduler"
        
        # Exécution du script Python avec les paramètres actuels
        python lab1_lea.py --batch_size $batch_size --epochs $epochs --lr $lr $scheduler
      done
    done
  done
done
