#!/bin/bash

# =================================================================
#  End-to-End Experiment Runner -- v3
#  - Trains the "v3" diffusion model on source code.
#  - Uses the fast DDIM sampler for sampling during and after training.
#  - Automatically runs inference using the final checkpoint.
# =================================================================

# Exit immediately if any command fails
set -e

# --- 1. Configuration ---
EPOCHS=50
# --- Select the GPUs to use ---
GPUS_TO_USE="1,2"
# --- Select the sampler to use ---
SAMPLER="ddim" # "ddim" or "ddpm"
DDIM_STEPS=50  # Only used if sampler is "ddim"
# --- Set the data source ---
DATA_DIR="$HOME/go/src/github.com/theapemachine"
CHECKPOINT_DIR="checkpoints"


# ================================================================
# Phase 1: Train the Model
# ================================================================
echo
echo "--- Phase 1: Training v3 model on code from '$DATA_DIR' for $EPOCHS epochs ---"
echo "This will take a while..."

# Calculate and set OMP_NUM_THREADS to avoid DDP warnings and improve performance.
# This logic divides available cores among the processes.
NUM_GPUS=$(echo $GPUS_TO_USE | awk -F, '{print NF}')
CPU_CORES=$(nproc)
THREADS_PER_PROC=$((CPU_CORES / NUM_GPUS))
export OMP_NUM_THREADS=$THREADS_PER_PROC

# Use torchrun for distributed training.
# It automatically manages the distributed environment.
CUDA_VISIBLE_DEVICES=$GPUS_TO_USE torchrun --nproc_per_node=$NUM_GPUS main.py \
    --phase train \
    --model "$MODEL_NAME" \
    --data-dir "$DATA_DIR" \
    --epochs "$EPOCHS" \
    --use-ema \
    --amp \
    --sampler "$SAMPLER" \
    --ddim-steps "$DDIM_STEPS" \
    --ckpt-dir "$CHECKPOINT_DIR"

echo -e "\n--- Training finished successfully! ---"


# --- 3. Inference Phase ---
echo -e "\n--- Phase 2: Generating samples from the last checkpoint ---"

# Find the latest checkpoint file in the directory.
LATEST_CHECKPOINT=$(ls -1 "$CHECKPOINT_DIR"/model_epoch_*.pt | sort -V | tail -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint file found in '$CHECKPOINT_DIR'."
    exit 1
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"

# Run the generation script with the final model.
# Inference is done on a single GPU.
FIRST_GPU=$(echo $GPUS_TO_USE | cut -d',' -f1)
CUDA_VISIBLE_DEVICES=$FIRST_GPU python generate.py \
    --checkpoint "$LATEST_CHECKPOINT" \
    --num-samples 5 \
    --sampler "$SAMPLER" \
    --ddim-steps "$DDIM_STEPS"

echo -e "\n--- Experiment finished! ---"