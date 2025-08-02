#!/usr/bin/env python3
# ================================================================
#  Diffusion-based Text Generator -- v3 Inference Script
# ================================================================

import argparse
from pathlib import Path

import torch

# --- Import all necessary components from the new main.py ---
from main import (
    DiffusionTransformerPE,
    DDPM_Sampler,
    DDIM_Sampler,
    get_noise_schedule_v2,
    set_seed,
    make_loader,
    hidden_size,
    sequence_length,
    num_timesteps,
    num_layers,
    num_heads,
    dim_feedforward,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text from a trained diffusion model.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    p.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate.")
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim", help="Which sampler to use.")
    p.add_argument("--ddim-steps", type=int, default=50, help="Number of steps for DDIM sampler.")
    p.add_argument("--ddim-eta", type=float, default=0.0, help="Eta value for DDIM sampler.")
    args = p.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Checkpoint and Vocab ---
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Handle old checkpoint format compatibility
    if "args" not in ckpt:
        raise ValueError(f"Checkpoint {ckpt_path} is missing 'args' key. "
                        f"This might be from an older format or corrupted. "
                        f"Available keys: {list(ckpt.keys())}")
    train_args = ckpt["args"]

    # Load cached vocabulary from checkpoint instead of rebuilding
    if "vocab" in ckpt:
        print("Loading cached vocabulary from checkpoint...")
        vocab = ckpt["vocab"]
    else:
        print("Warning: No cached vocab found. Rebuilding from dataset...")
        loader, _, _ = make_loader(
            batch_size=1,  # Only need a batch size of 1 for vocab
            seq_len=sequence_length,
            data_dir=train_args.data_dir,
        )
        vocab = loader.dataset.vocab
    pad_id = vocab["<pad>"]

    # --- 2. Reconstruct Model from Saved Args ---
    print("Reconstructing model from saved hyperparameters...")
    model = DiffusionTransformerPE(
        vocab_size=len(vocab),
        hidden=hidden_size,
        layers=num_layers,
        heads=num_heads,
        ff=dim_feedforward,
        max_len=sequence_length,
        pad_id=pad_id,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Model loaded successfully.")

    # --- 3. Instantiate Sampler ---
    alphas_cumprod = get_noise_schedule_v2(
        num_timesteps, schedule=train_args.schedule, device=device
    )

    if args.sampler == "ddim":
        print(f"Using DDIM sampler with {args.ddim_steps} steps.")
        sampler = DDIM_Sampler(
            model, alphas_cumprod, num_timesteps, device,
            steps=args.ddim_steps,
            eta=args.ddim_eta
        )
    else:
        print("Using DDPM sampler.")
        sampler = DDPM_Sampler(model, alphas_cumprod, num_timesteps, device)


    # --- 4. Generate Samples ---
    print(f"\n--- Generating {args.num_samples} samples using '{args.sampler.upper()}' sampler ---")
    for i in range(args.num_samples):
        # Generate a single sample
        sample_tokens = sampler.sample(B=1, L=sequence_length)
        sample_tokens = sample_tokens.squeeze(0)  # Remove batch dimension

        # Decode and print
        text_sample = " ".join(vocab.lookup_tokens(sample_tokens.tolist()))
        cleaned_text = text_sample.replace("<pad>", "").strip()

        print(f"\nSample {i + 1}:\n---\n{cleaned_text}\n---")


if __name__ == "__main__":
    main()