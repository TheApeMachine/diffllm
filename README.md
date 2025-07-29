# Diffusion-based Text/Code Generator

This project is an advanced, research-grade framework for training diffusion-based models for text and code generation using PyTorch. It has evolved from a simple proof-of-concept into a flexible, powerful tool incorporating several state-of-the-art techniques.

The model learns to reverse a "noising" process. It starts with pure Gaussian noise and iteratively refines it over hundreds of steps to produce a coherent sequence of embeddings, which are then decoded back into text or code tokens.

## Key Features

- **State-of-the-Art Architecture**: Uses a **Transformer Encoder** (the backbone of models like GPT and BERT) instead of a traditional LSTM, allowing for a much better understanding of long-range dependencies in the data.
- **Distributed Training**: Supports multi-GPU training out-of-the-box using PyTorch's `DistributedDataParallel` (DDP) and the `torchrun` launcher for significant speedups.
- **Advanced Training Techniques**:
    - **Cosine Noise Schedule**: Implements the improved noise schedule from Nichol & Dhariwal's "Improved DDPM" paper for potentially higher-quality results.
    - **Exponential Moving Average (EMA)**: Keeps a "shadow" copy of the model with smoothed weights, which often leads to more stable and better-performing samples.
    - **Gradient Accumulation**: Allows for training with effective batch sizes larger than what can fit in GPU memory.
    - **Mixed-Precision Training**: Uses `torch.cuda.amp` to accelerate training on modern NVIDIA GPUs.
- **Flexible Data Pipeline**:
    - Can train on the standard `IMDB` text dataset by default.
    - Includes a custom `CodeDataset` that can recursively scan a directory and train on any source code files (`.py`, `.go`, `.js`, etc.), allowing the model to learn code generation.
- **Modular and Usable**:
    - **Separate Training and Inference**: `main.py` handles training, while `generate.py` provides a clean, dedicated script for generating samples from saved checkpoints.
    - **Automated Experiment Runner**: `run_code_gen_experiment.sh` provides a one-command way to run a full end-to-end experiment: train the model and immediately generate samples from the final checkpoint.
    - **Self-Contained Checkpoints**: Model checkpoints save the training hyperparameters, allowing for easy and accurate model reconstruction during inference.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone diffllm
    cd diffllm
    ```

2.  **Create a Conda Environment (Recommended):**
    ```bash
    conda create -n diffusion_env python=3.10
    conda activate diffusion_env
    ```

3.  **Install Dependencies:**
    This project requires PyTorch and TorchText. The most reliable way to install them is to first uninstall any existing versions and then install compatible ones from the official PyTorch channel.

    ```bash
    # Uninstall any old versions first
    pip uninstall -y torch torchtext

    # Install compatible versions (for CUDA 12.1)
    # Visit https://pytorch.org/get-started/locally/ for other CUDA versions or CPU-only install.
    pip install --no-cache-dir torch torchtext --index-url https://download.pytorch.org/whl/cu121
    ```

## How to Use

### Training

The `main.py` script is the entry point for training. You can train on the default IMDB dataset or on a directory of code.

**Example: Train on the IMDB dataset (single GPU)**
```bash
python main.py --epochs 100 --use-ema --amp
```

**Example: Train on a local code directory (multi-GPU)**
The `run_code_gen_experiment.sh` script is pre-configured to handle this. Simply edit the `GPUS_TO_USE` and `DATA_DIR` variables in the script, then make it executable and run it.

```bash
# First, make the script executable
chmod +x run_code_gen_experiment.sh

# Run the end-to-end experiment
./run_code_gen_experiment.sh
```
This will train the model on the specified code and automatically run the generation script with the final checkpoint.

### Generating Samples

Once you have a trained checkpoint file (e.g., in the `checkpoints/` directory), you can use `generate.py` to create new samples on demand.

```bash
# Generate 5 samples from a specific checkpoint
python generate.py --checkpoint checkpoints/model_epoch_100.pt --num-samples 5
```
The script will automatically load the model with the correct hyperparameters that were saved in the checkpoint file.

## Command-Line Arguments

Both scripts support a number of arguments. Use the `-h` flag to see all options.

### `main.py` (Training)
- `--data-dir`: Path to a directory of code files. If not provided, uses the IMDB dataset.
- `--epochs`: Number of epochs to train for.
- `--schedule`: Noise schedule to use (`linear` or `cosine`).
- `--use-ema`: Enable Exponential Moving Average of model weights.
- `--amp`: Enable Automatic Mixed Precision for faster training.
- `--num-layers`, `--num-heads`, `--dim-ff`: Control the Transformer architecture.

### `generate.py` (Inference)
- `--checkpoint`: (Required) Path to the saved model checkpoint `.pt` file.
- `--num-samples`: Number of samples to generate.
- `--batch-size`: Batch size to use for generation.