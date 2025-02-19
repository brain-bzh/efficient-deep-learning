#!/bin/bash
tmux new -s training_session
python lab1_lea.py --batch_size 64 --epochs 50 --lr 0.001 --use_scheduler
