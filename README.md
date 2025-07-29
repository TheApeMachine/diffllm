# Diffusion-based Text/Code Generator (v3)

This project is an advanced, research-grade framework for training diffusion-based models for text and code generation using PyTorch. It has evolved from a simple proof-of-concept into a flexible, powerful tool incorporating several state-of-the-art techniques.

The model learns to reverse a "noising" process. It starts with pure Gaussian noise and iteratively refines it over hundreds of steps to produce a coherent sequence of embeddings, which are then decoded back into text or code tokens.

## Key Features

- **State-of-the-Art Architecture**:
    - **Transformer Backbone**: Uses a `TransformerEncoder` (the backbone of models like GPT and BERT) for a much better understanding of long-range dependencies in the data.
    - **Learned Positional Encodings**: Replaces static sinusoidal positional encodings with `nn.Embedding`-based learned positions for potentially better performance.
    - **Padding-Awareness**: The Transformer correctly ignores padding tokens in the input sequence, leading to more stable and accurate training.
- **Advanced Diffusion Techniques**:
    - **Self-Conditioning**: Improves sample quality by feeding the model's prediction of the clean data (`x0`) from the previous timestep back into the next one.
    - **DDIM Sampling**: Includes an implementation of Denoising Diffusion Implicit Models (DDIM), allowing for much faster sampling in as few as 50 steps instead of 1000, with comparable quality.
    - **Cosine Noise Schedule**: Implements the improved noise schedule from Nichol & Dhariwal's "Improved DDPM" paper for potentially higher-quality results.
- **Robust and Scalable Training**:
    - **Distributed Training**: Supports multi-GPU training out-of-the-box using PyTorch's `DistributedDataParallel` (DDP) and the `torchrun` launcher for significant speedups.
    - **Exponential Moving Average (EMA)**: Keeps a "shadow" copy of the model with smoothed weights, which often leads to more stable and better-performing samples.
    - **Gradient Accumulation**: Allows for training with effective batch sizes larger than what can fit in GPU memory.
    - **Mixed-Precision Training**: Uses `torch.cuda.amp` to accelerate training on modern NVIDIA GPUs.
- **Flexible Data Pipeline**:
    - Can train on the standard `IMDB` text dataset by default.
    - Includes a custom `CodeDataset` that can recursively scan a directory and train on any source code files (`.py`, `.go`, `.js`, etc.), allowing the model to learn code generation.
- **Modular and Usable**:
    - **Phased Execution**: `main.py` now handles both training (`--phase train`) and generation (`--phase generate`), providing a single entry point.
    - **Automated Experiment Runner**: `run_code_gen_experiment.sh` provides a one-command way to run a full end-to-end experiment: train the model and immediately generate samples from the final checkpoint.
    - **Self-Contained Checkpoints**: Model checkpoints save the training hyperparameters, allowing for easy and accurate model reconstruction during inference.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/theapemachine/diffllm.git
    cd diffllm
    ```

2.  **Create a Conda Environment (Recommended):**
    ```bash
    conda create -n diffusion_env python=3.10
    conda activate diffusion_env
    ```

3.  **Install Dependencies:**
    This project requires PyTorch and TorchText.

    ```bash
    # For CUDA 12.1 - visit https://pytorch.org/get-started/locally/ for other versions
    pip install torch torchtext --index-url https://download.pytorch.org/whl/cu121
    ```

## How to Use

### Training

The `main.py` script is the entry point for training. You can train on the default IMDB dataset or on a directory of code.

**Example: Train on IMDB using DDIM sampler (single GPU)**
```bash
python main.py --epochs 50 --use-ema --amp --sampler ddim --ddim-steps 100
```

**Example: Train on a local code directory (multi-GPU)**
The `run_code_gen_experiment.sh` script is pre-configured for this. Simply edit the `DATA_DIR` variable in the script, then make it executable and run it.

```bash
# First, make the script executable
chmod +x run_code_gen_experiment.sh

# Run the end-to-end experiment
./run_code_gen_experiment.sh
```
This will train the model on the specified code and save checkpoints to the `checkpoints/` directory.

### Generating Samples

Once you have a trained checkpoint, use the `generate.py` script to create new samples.

```bash
# Generate 5 samples from a specific checkpoint
python generate.py --checkpoint checkpoints/model_epoch_100.pt --num-samples 5
```
The script automatically loads the model with the correct hyperparameters saved in the checkpoint file.

## Command-Line Arguments

The `main.py` script supports a number of arguments. Use the `-h` flag to see all options.

- `--data-dir`: Path to a directory of code files. If not provided, uses the IMDB dataset.
- `--phase`: Set to `train` or `generate`.
- `--epochs`: Number of epochs to train for.
- `--learning_rate`: The learning rate for the Adam optimizer.
- `--schedule`: Noise schedule to use (`linear` or `cosine`).
- `--sampler`: Sampler to use for generating images during training (`ddpm` or `ddim`).
- `--ddim-steps`: Number of steps for the DDIM sampler.
- `--grad-clip`: Maximum norm for gradient clipping.
- `--grad-accum`: Number of steps to accumulate gradients over.
- `--use-ema`: Enable Exponential Moving Average of model weights.
- `--amp`: Enable Automatic Mixed Precision for faster training.
- `--model`: Name of the model and tokenizer files.
- `--ckpt-dir`: Directory to save checkpoints.
