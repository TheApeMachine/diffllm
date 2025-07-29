#!/usr/bin/env python3
# ================================================================
#  Diffusion‑based Code‑Generator — extended edition  ✧ v2 ✧
#  (original lines retained; see “NEW CAPABILITIES” section below)
# ================================================================

import math
import os
import random
import argparse
import time
import inspect
from pathlib import Path
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab

# ----------------------------------------------------------------
# 0. Reproducibility helpers
# ----------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Make results (reasonably) reproducible across CPU & GPU runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # slower, but reproducible
    torch.backends.cudnn.benchmark = False

set_seed()

# ----------------------------------------------------------------
# Original code ‑‑ kept verbatim
# ----------------------------------------------------------------
# NOTE: Global `device` variable is removed. It will be managed by the main() function.

# --- 1. Timestep Embedding ---
# This module converts a timestep integer into a continuous, high-dimensional vector.
# This allows the model to know at which point in the diffusion process it is operating.
class TimestepEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

# --- 2. The Diffusion Model ---
# This is the core neural network that learns to reverse the diffusion process.
# It's conditioned on the noisy input and the current timestep.
class DiffusionModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dim_feedforward):
        super(DiffusionModel, self).__init__()
        # The embedding layer is part of the model, its weights will be trained
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Timestep embedding to condition the model on the noise level
        self.time_embedding = TimestepEmbedding(hidden_size)
        
        # --- The Transformer Encoder ---
        # This replaces the LSTM. It's more powerful for capturing context.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True # This is crucial for our data shape
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # The linear layer predicts the noise that was added to the embeddings
        # Its output size must be hidden_size to match the noise dimension
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, noisy_embeddings, t):
        # noisy_embeddings: (batch, seq_len, hidden_size)
        # t: (batch,) with timesteps for each sequence
        
        # 1. Get timestep embedding and add it to the input
        time_emb = self.time_embedding(t).unsqueeze(1)
        conditioned_input = noisy_embeddings + time_emb
        
        # 2. Pass through the Transformer Encoder
        transformer_out = self.transformer_encoder(conditioned_input)
        
        # 3. Predict the added noise from the Transformer's output
        noise_prediction = self.linear(transformer_out)
        
        return noise_prediction

# --- 3. Hyperparameters ---
vocab_size = 10000
hidden_size = 512
num_layers = 6 # Transformers can be deeper
learning_rate = 0.001
batch_size = 32
sequence_length = 64
num_timesteps = 1000  # A more realistic number of timesteps for diffusion
epochs = 100 # Increased epochs for demonstration
steps_per_epoch = 200 # Run multiple steps per "epoch" for more frequent feedback
# --- New Transformer Hyperparameters ---
num_heads = 8 # Number of attention heads
dim_feedforward = 2048 # Dimension of the feedforward network

# --- 4. Noise Schedule ---
# Defines how much noise is added at each timestep.
# This uses a linear schedule for the variance (beta).
def get_noise_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    return alphas_cumprod

# Helper to get the correct alpha values for a batch of timesteps
def get_schedule_values(alphas_cumprod, t, shape):
    batch_size = t.shape[0]
    alphas_t = alphas_cumprod.gather(-1, t)
    return alphas_t.reshape(batch_size, *((1,) * (len(shape) - 1))).expand(shape)

# --- 5. Training Loop ---
def train(model, criterion, optimizer, epochs, steps_per_epoch):
    alphas_cumprod = get_noise_schedule(num_timesteps)
    
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            # --- The Diffusion Process ---
            
            # Start with clean data (e.g., a batch of token sequences)
            inputs = torch.randint(0, vocab_size, (batch_size, sequence_length), device=device)
            
            # Sample a random timestep t for each sequence in the batch
            t = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
            
            # 1. Convert token IDs to continuous embeddings
            clean_embeddings = model.embedding(inputs)
            
            # 2. Get noise and alpha values for the sampled timesteps
            sqrt_alphas_cumprod_t = get_schedule_values(
                torch.sqrt(alphas_cumprod), t, clean_embeddings.shape
            )
            sqrt_one_minus_alphas_cumprod_t = get_schedule_values(
                torch.sqrt(1. - alphas_cumprod), t, clean_embeddings.shape
            )
            
            # 3. Create noise and apply it to the embeddings to get x_t
            noise = torch.randn_like(clean_embeddings)
            noisy_embeddings = sqrt_alphas_cumprod_t * clean_embeddings + sqrt_one_minus_alphas_cumprod_t * noise
            
            # --- The Denoising (Reverse) Process ---
            
            # 4. Predict the original noise from the noisy embeddings
            noise_prediction = model(noisy_embeddings, t)
            
            # 5. Calculate the loss between the model's prediction and the actual noise
            loss = criterion(noise_prediction, noise)
            
            # 6. Backward pass and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f'Epoch [{epoch+1}/{epochs}], Last Step Loss: {loss.item():.4f}')

# ----------------------------------------------------------------
# ======================  NEW  CAPABILITIES  ======================
# ----------------------------------------------------------------
#  Everything beyond this point is *additive*.  Older imports / names
#  are untouched, so your previous workflows keep working.

# --- 6‑A. Positional Encoding & Mask‑aware Transformer -----------------------
class PositionalEncoding(nn.Module):
    """Learned positional embeddings (better than sinusoid for code tokens)."""
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        B, L, _ = x.size()
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        return self.pe(pos)


class DiffusionTransformerPE(nn.Module):
    """
    Improved model featuring:
      • learned positional embeddings
      • pad‑masking in attention
      • same public API as DiffusionModel, plus optional pad_mask arg
    """
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 num_layers: int,
                 num_heads: int,
                 dim_feedforward: int,
                 max_seq_len: int,
                 pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_id)
        self.pos_embedding = PositionalEncoding(max_seq_len, hidden_size)
        self.time_embedding = TimestepEmbedding(hidden_size)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self,
                noisy_embeddings: torch.Tensor,
                t: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        noisy_embeddings : (B, L, H)
        t                : (B,)
        pad_mask         : (B, L) True for PAD tokens; may be None
        """
        time_emb = self.time_embedding(t).unsqueeze(1)            # (B,1,H)
        x = noisy_embeddings + time_emb + self.pos_embedding(noisy_embeddings)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.proj(x)                                        # ε̂

# --- 6‑B. β‑schedule variants (linear | cosine) ------------------------------
def get_noise_schedule_v2(num_timesteps: int,
                          schedule: str = "linear",
                          beta_start: float = 0.0001,
                          beta_end: float = 0.02,
                          device: torch.device = torch.device("cpu"),
                          ) -> torch.Tensor:
    """
    Return ᾱ_t (cumprod) for chosen schedule.
    Ref: Improved DDPM (Nichol & Dhariwal).
    """
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
    elif schedule == "cosine":
        s = 0.008  # offset per paper
        steps = torch.arange(num_timesteps + 1, device=device) / num_timesteps
        alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(1e-5, 0.999)
    else:
        raise ValueError(f"Unknown schedule '{schedule}'")
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)

# --- 6‑C. Exponential Moving Average (EMA) of weights ------------------------
class EMA:
    """Shadow parameters for more stable sampling."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

# --- 6‑D. Utility helpers ---------------------------------------------------
def embeddings_to_tokens(
    embeds: torch.Tensor,  # (B, L, H)
    embedding_weights: torch.Tensor
) -> torch.Tensor:
    """Nearest‑neighbor decode embeddings back to token IDs."""
    logits = torch.matmul(embeds, embedding_weights.t())  # (B, L, |V|)
    return logits.argmax(dim=-1)

# --- 6‑E. Diffusion Sampler  (unchanged logic, now EMA‑aware) ---------------
class DiffusionSampler(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        alphas_cumprod: torch.Tensor,
        num_timesteps: int,
        device: torch.device,
    ):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device

        acp = alphas_cumprod.to(device)
        acp_prev = torch.cat([torch.tensor([1.0], device=device), acp[:-1]])
        alphas = acp / acp_prev
        betas = 1.0 - alphas

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - acp))

    @torch.no_grad()
    def sample(self,
               batch_size: int,
               seq_len: int,
               pad_mask: Optional[torch.Tensor] = None,
               return_tokens: bool = True
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_t = torch.randn(batch_size, seq_len, hidden_size, device=self.device)

        for t_ind in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_ind, dtype=torch.long, device=self.device)
            # Model may or may not take a pad_mask argument
            sig = inspect.signature(self.model.forward)
            if "pad_mask" in sig.parameters:
                eps_theta = self.model(x_t, t, pad_mask)
            else:
                eps_theta = self.model(x_t, t)

            posterior_mean = self.sqrt_recip_alphas[t_ind] * (
                x_t - (self.betas[t_ind] / self.sqrt_one_minus_alphas_cumprod[t_ind]) * eps_theta
            )

            if t_ind > 0:
                noise = torch.randn_like(x_t)
                x_t = posterior_mean + torch.sqrt(self.betas[t_ind]) * noise
            else:
                x_t = posterior_mean

        if return_tokens:
            tokens = embeddings_to_tokens(x_t, self.model.embedding.weight)
            return x_t, tokens
        return x_t

# --- 6‑F. Gradient‑accumulating, EMA‑aware Trainer --------------------------
def train_improved(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
    grad_clip: float,
    grad_accum: int,
    ckpt_dir: str,
    amp: bool,
    pad_id: int,
    alphas_cumprod: torch.Tensor,
    train_loader: DataLoader,
    vocab,
    ema: Optional[EMA],
    sample_every: int,
    args: argparse.Namespace,
    device: torch.device,
    is_ddp: bool,
) -> None:

    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(exist_ok=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    # Determine rank for distributed training
    rank = dist.get_rank() if is_ddp else 0

    data_iter = iter(train_loader)
    running_loss = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        for step in range(steps_per_epoch):
            # ---------------- Load batch ----------------
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            inputs = batch.to(device)           # (B, L)
            pad_mask = (inputs == pad_id)       # (B, L)

            # ---------------- Forward diffusion ---------
            B = inputs.size(0)
            t = torch.randint(0, num_timesteps, (B,), device=device, dtype=torch.long)

            clean_emb = model.embedding(inputs)
            sqrt_a = get_schedule_values(torch.sqrt(alphas_cumprod), t, clean_emb.shape)
            sqrt_1ma = get_schedule_values(torch.sqrt(1.0 - alphas_cumprod), t, clean_emb.shape)
            noise = torch.randn_like(clean_emb)
            noisy_emb = sqrt_a * clean_emb + sqrt_1ma * noise

            # ---------------- Reverse + loss ------------
            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                sig = inspect.signature(model.forward)
                if "pad_mask" in sig.parameters:
                    pred_noise = model(noisy_emb, t, pad_mask)
                else:
                    pred_noise = model(noisy_emb, t)
                loss = criterion(pred_noise, noise) / grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum  # un‑scaled for logging

            # ------------- Optimise every grad_accum ----
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.update(model)

        # ---------------- Epoch‑end log / sample -------
        avg_loss = running_loss / steps_per_epoch
        running_loss = 0.0
        dt = time.time() - t0
        print(f"Epoch {epoch:03}/{epochs:03}  loss={avg_loss:.4f}  {dt:.1f}s")

        # Checkpoint + quick text sample (only on main process)
        if rank == 0 and ((epoch % sample_every == 0) or (epoch == epochs)):
            # When using DDP, we need to save the state of the underlying model.
            model_to_save = model.module if is_ddp else model

            ckpt_file = ckpt_path / f"model_epoch{epoch:03}.pt"
            torch.save(
                {"epoch": epoch,
                 "model_state": model_to_save.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "args": args}, # <-- Save training args
                ckpt_file
            )
            print(f" ↳ checkpoint saved: {ckpt_file}")

            # Sampling (with EMA weights if available)
            sampler = DiffusionSampler(model_to_save, alphas_cumprod, num_timesteps, device)
            if ema:
                ema.apply_to(model_to_save)
            model_to_save.eval()
            _, sample_tok = sampler.sample(batch_size=1, seq_len=inputs.size(1))
            if ema:
                ema.restore(model_to_save)
            
            # Switch back to train mode for the main model
            model.train()

            text = " ".join(vocab.lookup_tokens(sample_tok.squeeze(0).tolist()))
            print(f' ↳ sample: "{text}"')

# --- 6‑G. Code‑corpus Dataset (optional) ------------------------------------
class CodeDataset(Dataset):
    """Simple recursive loader turning code files into token sequences."""
    def __init__(self, root_dir: str, tokenizer, seq_len: int):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.samples: List[List[str]] = []

        root = Path(root_dir)
        print(f"Scanning {root} for code files …")
        for path in root.rglob("*"):
            if path.suffix.lower() in {".py", ".js", ".ts", ".go", ".java", ".cs", ".cpp", ".c"}:
                try:
                    text = path.read_text(encoding="utf8", errors="ignore")
                except Exception:
                    continue
                tokens = tokenizer(text)
                # Slice long files into many seq_len chunks
                for i in range(0, len(tokens), seq_len):
                    chunk = tokens[i:i + seq_len]
                    if chunk:
                        self.samples.append(chunk)

        if not self.samples:
            raise RuntimeError(f"No suitable code files found in {root_dir}")
        print(f"CodeDataset: {len(self.samples)} chunks.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def build_vocab_from_dataset(dataset: List[List[str]],
                             specials: List[str]) -> Vocab:
    return build_vocab_from_iterator(iter(dataset), specials=specials)

def code_collate(batch: List[List[str]],
                 vocab,
                 seq_len: int,
                 pad_id: int) -> torch.Tensor:
    out = []
    for tokens in batch:
        ids = vocab(tokens)
        if len(ids) < seq_len:
            ids += [pad_id] * (seq_len - len(ids))
        else:
            ids = ids[:seq_len]
        out.append(torch.tensor(ids, dtype=torch.long))
    return torch.stack(out)

# --- 6‑H. Data loading helper -----------------------------------------------
def prepare_data_loader(
    batch_size: int,
    seq_len: int,
    data_dir: Optional[str] = None,
    is_ddp: bool = False,
):
    tokenizer = get_tokenizer("basic_english")

    specials = ["<unk>", "<pad>"]
    pad_token = "<pad>"

    if data_dir is None:
        print("Preparing IMDB dataset …")
        train_iter = IMDB(split="train")

        def yield_tokens(data_iter):
            for _, txt in data_iter:
                yield tokenizer(txt)

        vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=specials)
        vocab.set_default_index(vocab["<unk>"])

        train_iter = IMDB(split="train")  # reload
        dataset = [(lbl, txt) for lbl, txt in train_iter if txt.strip()]

        sampler = DistributedSampler(dataset) if is_ddp else None
        def imdb_collate(batch):
            texts = []
            for _, txt in batch:
                tokens = tokenizer(txt)
                ids = vocab(tokens)
                if len(ids) < seq_len:
                    ids += [vocab[pad_token]] * (seq_len - len(ids))
                else:
                    ids = ids[:seq_len]
                texts.append(torch.tensor(ids, dtype=torch.long))
            return torch.stack(texts)

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=(sampler is None), # Shuffle must be False with a sampler
                            sampler=sampler,
                            collate_fn=imdb_collate,
                            drop_last=True)
    else:
        code_ds = CodeDataset(data_dir, tokenizer, seq_len)
        vocab = build_vocab_from_dataset(code_ds.samples, specials=specials)
        vocab.set_default_index(vocab["<unk>"])
        pad_id = vocab[pad_token]
        
        sampler = DistributedSampler(code_ds) if is_ddp else None

        loader = DataLoader(code_ds,
                            batch_size=batch_size,
                            shuffle=(sampler is None), # Shuffle must be False with a sampler
                            sampler=sampler,
                            collate_fn=lambda b: code_collate(b, vocab, seq_len, pad_id),
                            drop_last=True)

    print(f"Vocab size: {len(vocab)}")
    return loader, vocab, vocab[pad_token]

# --- 6‑I. CLI & entry‑point --------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a diffusion model for (code) text generation.")
    parser.add_argument("--epochs", type=int, default=epochs)
    parser.add_argument("--steps", type=int, default=steps_per_epoch)
    parser.add_argument("--lr", type=float, default=learning_rate)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root directory containing code files. "
                             "If omitted, IMDB text dataset is used.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-layers", type=int, default=num_layers)
    parser.add_argument("--num-heads", type=int, default=num_heads)
    parser.add_argument("--dim-ff", type=int, default=dim_feedforward)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--sample-every", type=int, default=10,
                        help="Save CKPT & sample every N epochs")
    parser.add_argument("--use-ema", action="store_true", help="Enable EMA tracking")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    args = parser.parse_args()

    # --- DDP Setup ---
    is_ddp = 'WORLD_SIZE' in os.environ
    if is_ddp:
        # These are set by torchrun
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group("nccl")
        # The device for this process is the local rank
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        print(f"[Process {rank}] Initialized on device {device}.")
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on single device: {device}")


    # ---------- Data ----------
    loader, vocab, pad_id = prepare_data_loader(
        batch_size, sequence_length, args.data_dir, is_ddp=is_ddp
    )
    vocab_size_actual = len(vocab)

    # ---------- Model ----------
    model = DiffusionTransformerPE(
        vocab_size=vocab_size_actual,
        hidden_size=hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_ff,
        max_seq_len=sequence_length,
        pad_id=pad_id
    ).to(device)

    # --- Wrap model for DDP ---
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
        print(f"[Process {rank}] Wrapped model with DDP.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    alphas_cumprod = get_noise_schedule_v2(
        num_timesteps, schedule=args.schedule, device=device
    )

    # ---------- EMA ----------
    # EMA should only track on the main process
    if rank == 0 and args.use_ema:
        ema = EMA(model.module if is_ddp else model, decay=args.ema_decay)
    else:
        ema = None

    # ---------- Resume? ----------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        # When loading, we load into the unwrapped model
        model_to_load = model.module if is_ddp else model
        model_to_load.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    # ---------- Train ----------
    # Only the main process should print headers
    if rank == 0:
        print("\n--- Starting Training ---")

    train_improved(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        epochs=args.epochs,
        steps_per_epoch=args.steps,
        grad_clip=args.grad_clip,
        grad_accum=args.grad_accum,
        ckpt_dir="checkpoints",
        amp=args.amp,
        pad_id=pad_id,
        alphas_cumprod=alphas_cumprod,
        train_loader=loader,
        vocab=vocab,
        ema=ema,
        sample_every=args.sample_every,
        args=args, # Pass args to be saved in checkpoint
        device=device,
        is_ddp=is_ddp,
    )
    
    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
