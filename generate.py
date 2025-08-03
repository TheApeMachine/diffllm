
#!/usr/bin/env python3
# ================================================================
#  Diffusion-based Text Generator -- v3 Inference Script
# ================================================================

import argparse
from pathlib import Path
import torch
from tokenizers import Tokenizer

# --- Import all necessary components from the new main.py ---
from main import (
    DiffusionTransformerPE,
    DDPM_Sampler,
    DDIM_Sampler,
    get_noise_schedule_v2,
    set_seed,
    num_timesteps, # Keep this default
)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text from a trained diffusion model.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    p.add_argument("--num-samples", type=int, default=5, help="Number of samples to generate.")
    p.add_argument("--sampler", choices=["ddpm", "ddim"], default="ddim", help="Which sampler to use.")
    p.add_argument("--ddim-steps", type=int, default=50, help="Number of steps for DDIM sampler.")
    p.add_argument("--ddim-eta", type=float, default=0.0, help="Eta value for DDIM sampler.")
    p.add_argument("--sequence-length", type=int, default=None, help="Override sequence length from training.")
    p.add_argument("--prompt", type=str, default=None, help="Prompt for conditional generation.")
    p.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale for CFG (e.g., 0.0 for unconditional, 7.5 for strong guidance).")
    args = p.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Load Checkpoint and Tokenizer ---
    print(f"Loading checkpoint from: {args.checkpoint}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    
    train_args = ckpt.get("args")
    if not train_args:
        raise ValueError("Checkpoint must contain 'args' key from training.")

    tokenizer_path = ckpt.get("tokenizer_path")
    if not tokenizer_path or not Path(tokenizer_path).exists():
        raise ValueError(f"Tokenizer path not found in checkpoint or file missing: {tokenizer_path}")
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None:
        raise ValueError("Tokenizer must have a <pad> token.")

    # --- 2. Reconstruct Model from Saved Args ---
    print("Reconstructing model from saved hyperparameters...")
    seq_len = args.sequence_length if args.sequence_length else train_args.sequence_length
    
    model = DiffusionTransformerPE(
        vocab_size=tokenizer.get_vocab_size(),
        hidden=train_args.hidden_size,
        layers=train_args.num_layers,
        heads=train_args.num_heads,
        ff=train_args.dim_feedforward,
        max_len=seq_len,
        pad_id=pad_id,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Model loaded successfully.")

    # --- 3. Prepare Prompt Embedding ---
    prompt_emb = None
    if args.prompt:
        print(f"Using prompt for conditional generation: '{args.prompt}'")
        prompt_tokens = tokenizer.encode(args.prompt).ids
        if len(prompt_tokens) < seq_len:
            prompt_tokens += [pad_id] * (seq_len - len(prompt_tokens))
        else:
            prompt_tokens = prompt_tokens[:seq_len]
        
        prompt_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
        prompt_emb = model.embedding(prompt_tensor)
        prompt_emb = prompt_emb.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
    else:
        print("Running in unconditional mode (no prompt provided).")


    # --- 4. Instantiate Sampler ---
    alphas_cumprod = get_noise_schedule_v2(
        num_timesteps, schedule=train_args.schedule, device=device
    )

    sampler_args = {"model": model, "acp": alphas_cumprod, "T": num_timesteps, "device": device, "hidden_size": train_args.hidden_size}
    if args.sampler == "ddim":
        print(f"Using DDIM sampler with {args.ddim_steps} steps.")
        sampler = DDIM_Sampler(**sampler_args, steps=args.ddim_steps, eta=args.ddim_eta)
    else:
        print("Using DDPM sampler.")
        sampler = DDPM_Sampler(**sampler_args)


    # --- 5. Generate Samples ---
    print(f"\n--- Generating {args.num_samples} samples (guidance_scale={args.guidance_scale}) ---")
    for i in range(args.num_samples):
        sample_tokens = sampler.sample(
            B=1, 
            L=seq_en, 
            prompt_emb=prompt_emb, 
            guidance_scale=args.guidance_scale
        )
        sample_tokens = sample_tokens.squeeze(0)

        text_sample = tokenizer.decode(sample_tokens.tolist(), skip_special_tokens=True)

        print(f"\nSample {i + 1}:\n---\n{text_sample}\n---")


if __name__ == "__main__":
    main()
