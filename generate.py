#!/usr/bin/env python3
# ================================================================
#  Diffusion-based Text Generator -- Inference Script
# ================================================================

import argparse
from pathlib import Path

import torch

# --- We need to import the model and sampler definitions from main.py ---
# This is a common practice for separating training and inference code.
from main import (
    DiffusionTransformerPE,
    DiffusionSampler,
    get_noise_schedule_v2,
    set_seed,
    device,
    prepare_data_loader,
    hidden_size,       # Use some defaults from main
    sequence_length,
    num_timesteps,
    batch_size,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate text samples from a trained diffusion model."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint (.pt) file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of independent samples to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        help="Batch size for generation (can be larger than training batch size).",
    )
    args = parser.parse_args()

    set_seed()

    # --- 1. Load Checkpoint and Vocab ---
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    train_args = ckpt["args"]

    # The vocabulary depends on the dataset used during training.
    # We re-run prepare_data_loader to get the correct vocab.
    print("Rebuilding vocabulary from original dataset...")
    _, vocab, pad_id = prepare_data_loader(
        batch_size=args.batch_size,
        seq_len=sequence_length,
        data_dir=train_args.data_dir,
    )

    # --- 2. Reconstruct the Model Architecture ---
    print("Reconstructing model from saved hyperparameters...")
    model = DiffusionTransformerPE(
        vocab_size=len(vocab),
        hidden_size=hidden_size,
        num_layers=train_args.num_layers,
        num_heads=train_args.num_heads,
        dim_feedforward=train_args.dim_ff,
        max_seq_len=sequence_length,
        pad_id=pad_id,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Model loaded successfully.")

    # --- 3. Generate Samples ---
    alphas_cumprod = get_noise_schedule_v2(
        num_timesteps, schedule=train_args.schedule
    )
    sampler = DiffusionSampler(model, alphas_cumprod, num_timesteps)

    print(f"\n--- Generating {args.num_samples} samples ---")
    for i in range(args.num_samples):
        # Generate a batch of samples
        _, sample_tokens = sampler.sample(
            batch_size=1,  # Generate one sample at a time
            seq_len=sequence_length,
            return_tokens=True,
        )

        # Decode the tokens into human-readable text
        token_list = sample_tokens.squeeze().tolist()
        text_sample = " ".join(vocab.lookup_tokens(token_list))

        # Clean up the output for readability
        cleaned_text = text_sample.replace("<pad>", "").strip()
        print(f"\nSample {i + 1}:\n---\n{cleaned_text}\n---")


if __name__ == "__main__":
    main()