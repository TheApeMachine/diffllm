#!/bin/bash

# =================================================================
#  End-to-End Experiment Runner
#  - Trains the diffusion model on source code in this project.
#  - Automatically runs inference using the final checkpoint.
# =================================================================

# Exit immediately if any command fails
set -e

# --- 1. Configuration ---
# You can change these parameters to control the experiment.
EPOCHS=50
SAMPLE_EVERY=10
# --- Select the GPUs to use ---
# Set this to "0,1,2" to use three GPUs, or "0" for a single GPU.
# The number of processes will be set automatically based on this list.
GPUS_TO_USE="1,2"
CHECKPOINT_DIR="checkpoints"
# Use the current directory "." as the source for code files.
# The model will train on its own source code!
# DATA_DIR="."
DATA_DIR="$HOME/go/src/github.com/theapemachine"


# --- 2. Training Phase ---
echo -e "\n--- Phase 1: Training model on code from '$DATA_DIR' for $EPOCHS epochs ---"
echo "This will take a while..."

# Use torchrun for distributed training.
# It automatically sets environment variables for DDP.
# --nproc_per_node is derived from the number of GPUs in the list.
NUM_GPUS=$(echo $GPUS_TO_USE | tr ',' ' ' | wc -w)

CUDA_VISIBLE_DEVICES=$GPUS_TO_USE torchrun --nproc_per_node=$NUM_GPUS main.py \
    --data-dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --sample-every "$SAMPLE_EVERY" \
    --use-ema \
    --amp

echo -e "\n--- Training finished successfully! ---"


# --- 3. Inference Phase ---
echo -e "\n--- Phase 2: Generating samples from the last checkpoint ---"

# Find the latest checkpoint file in the directory.
# This assumes the highest epoch number is the latest.
LATEST_CHECKPOINT=$(ls -1 "$CHECKPOINT_DIR"/model_epoch_*.pt | sort -V | tail -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint file found in '$CHECKPOINT_DIR'."
    echo "Please ensure training completed and saved a checkpoint."
    exit 1
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"

# Run the generation script with the final model.
# Inference is done on a single GPU. We'll use the first one from the list.
FIRST_GPU=$(echo $GPUS_TO_USE | cut -d',' -f1)
CUDA_VISIBLE_DEVICES=$FIRST_GPU python generate.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --num-samples 5

echo -e "\n--- Experiment finished! ---"