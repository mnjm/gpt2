#!/bin/bash

LOG_DT=$(date '+%Y-%m-%d_%H-%M-%S')
LOGFILE="run_log_$LOG_DT.txt"

NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

SCRIPT="train.py"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Detected $NUM_GPUS GPU(s). Using torchrun..."
    torchrun --standalone --nproc_per_node=$NUM_GPUS $SCRIPT 2>&1 | tee "$LOGFILE"
else
    python $SCRIPT $ARGS 2>&1 | tee "$LOGFILE"
fi