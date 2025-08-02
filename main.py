#!/usr/bin/env python3
# ================================================================
#  Diffusion‑based Code‑Generator — extended edition  ✧ v3 ✧
# ================================================================

import math
import os
import random
import re
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
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ----------------------------------------------------------------
# ----------------------- ORIGINAL   CODE ------------------------
# ----------------------------------------------------------------
# (unchanged except where explicitly noted)

# --- 1. Timestep Embedding ---
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
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

# --- 2. Hyper‑parameters (defaults) ---
vocab_size = 10000
hidden_size = 512
num_layers = 6
learning_rate = 1e-3
batch_size = 32
sequence_length = 64
num_timesteps = 1000
epochs = 100
steps_per_epoch = 200
num_heads = 8
dim_feedforward = 2048

# ----------------------------------------------------------------
# ======================  NEW  CAPABILITIES  =====================
# ----------------------------------------------------------------
#  Everything beyond this line is additive / replace‑in‑place.

# 6‑A. Learned positional encodings + padding‑aware Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.size()
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        return self.pe(pos)

class DiffusionTransformerPE(nn.Module):
    """
    * self‑conditioning ready
    * returns (noise_pred, x0_pred)
    """
    def __init__(self, vocab_size: int, hidden: int, layers: int,
                 heads: int, ff: int, max_len: int, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_id)
        self.pos_embedding = PositionalEncoding(max_len, hidden)
        self.time_embedding = TimestepEmbedding(hidden)
        self.sc_proj = nn.Linear(hidden, hidden, bias=False)      # self‑cond projection

        enc = nn.TransformerEncoderLayer(d_model=hidden,
                                         nhead=heads,
                                         dim_feedforward=ff,
                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(hidden, hidden * 2)                 # outputs ε̂  |  x̂₀

    def forward(self, noisy_emb: torch.Tensor, t: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None,
                self_cond: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self_cond is None:
            self_cond = torch.zeros_like(noisy_emb)
        x = noisy_emb + self.sc_proj(self_cond)
        x = x + self.pos_embedding(x) + self.time_embedding(t).unsqueeze(1)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        noise_pred, x0_pred = self.head(x).chunk(2, dim=-1)
        return noise_pred, x0_pred

# 6‑B. β‑schedule (linear | cosine)
def get_noise_schedule_v2(T: int, schedule: str, device, beta_start=1e-4, beta_end=2e-2):
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, T, device=device)
    elif schedule == "cosine":
        s = 0.008
        steps = torch.arange(T + 1, device=device) / T
        al_c = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        al_c = al_c / al_c[0]
        betas = 1 - (al_c[1:] / al_c[:-1])
        betas = betas.clamp(1e-5, .999)
    else:
        raise ValueError(schedule)
    alphas = 1 - betas
    return torch.cumprod(alphas, dim=0)

# 6‑C. EMA (unchanged)
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        # Register the model parameters, handling DDP-wrapped models
        model_to_register = model.module if hasattr(model, 'module') else model
        for n, p in model_to_register.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self, model):
        # Update shadow parameters, handling DDP-wrapped models
        model_to_update = model.module if hasattr(model, 'module') else model
        for n, p in model_to_update.named_parameters():
            if p.requires_grad:
                assert n in self.shadow, f"Parameter {n} not found in EMA shadow."
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_to(self, model: nn.Module):
        # Use in-place swap to avoid memory spike from cloning all parameters
        self.backup = {}
        model_to_apply = model.module if hasattr(model, 'module') else model
        for n, p in model_to_apply.named_parameters():
            if p.requires_grad and n in self.shadow:
                # Swap data pointers instead of cloning to save memory
                self.backup[n] = p.data
                p.data = self.shadow[n]

    def restore(self, model: nn.Module):
        # Restore using pointer swap (matches apply_to optimization)
        model_to_restore = model.module if hasattr(model, 'module') else model
        for n, p in model_to_restore.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n]

# 6‑D. helpers
def get_schedule_values(acp: torch.Tensor, t: torch.Tensor, shape):
    B = t.shape[0]
    al_t = acp.gather(0, t)
    return al_t.view(B, *((1,) * (len(shape) - 1))).expand(shape)

def embeddings_to_tokens(emb: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    # Use gumbel softmax with low temperature for differentiable sampling
    # This reduces OOV drift compared to hard argmax
    logits = torch.matmul(emb, W.t())
    # Use temperature of 0.1 for sharp but not completely hard selection
    return torch.nn.functional.gumbel_softmax(logits, tau=0.1, hard=True, dim=-1).argmax(-1)

# 6‑E. DDPM Sampler (self‑cond aware)
class DDPM_Sampler(nn.Module):
    def __init__(self, model, acp, T, device):
        super().__init__()
        self.model, self.acp, self.T, self.device = model, acp, T, device
        acp_prev = torch.cat([torch.tensor([1.], device=device), acp[:-1]])
        self.alphas, self.betas = acp / acp_prev, 1 - acp / acp_prev
        self.sqrt_recip = (1 / self.alphas).sqrt()
        self.sqrt_om_acp = (1 - acp).sqrt()

    @torch.no_grad()
    def sample(self, B, L, pad_mask=None):
        x = torch.randn(B, L, hidden_size, device=self.device)
        self_cond = None
        for t_ind in reversed(range(self.T)):
            t = torch.full((B,), t_ind, dtype=torch.long, device=self.device)
            eps, x0 = self.model(x, t, pad_mask, self_cond)
            self_cond = x0                      # feed self‑cond next step
            mean = self.sqrt_recip[t_ind] * (x - self.betas[t_ind] / self.sqrt_om_acp[t_ind] * eps)
            if t_ind > 0:
                noise = torch.randn_like(x)
                x = mean + self.betas[t_ind].sqrt() * noise
            else:
                x = mean
        tok = embeddings_to_tokens(x, self.model.embedding.weight)
        return tok

# 6‑F. DDIM Sampler ----------------------------------------------------------
class DDIM_Sampler(nn.Module):
    def __init__(self, model, acp, T, device, steps=50, eta=0.0):
        super().__init__()
        self.model, self.device = model, device
        self.acp = acp
        self.eta = eta
        self.steps = torch.linspace(0, T - 1, steps, dtype=torch.long)

    @torch.no_grad()
    def sample(self, B, L, pad_mask=None):
        x = torch.randn(B, L, hidden_size, device=self.device)
        self_cond = None
        for i in reversed(range(1, len(self.steps))):
            t = self.steps[i]
            t_prev = self.steps[i - 1]
            a_t, a_prev = self.acp[t], self.acp[t_prev]

            t_long = torch.full((B,), t.item(), dtype=torch.long, device=self.device)
            eps, x0 = self.model(x, t_long, pad_mask, self_cond)
            self_cond = x0

            x0_pred = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            sigma = self.eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            dir_xt = (1 - a_prev - sigma ** 2).sqrt() * eps
            noise = torch.randn_like(x) if sigma.item() > 0 else 0
            x = a_prev.sqrt() * x0_pred + dir_xt + sigma * noise

        tok = embeddings_to_tokens(x, self.model.embedding.weight)
        return tok

# 6‑F. Checkpoint utilities -----------------------------------------------
def find_latest_checkpoint(ckpt_dir: str) -> Optional[Path]:
    """Find the latest checkpoint file by epoch number."""
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists():
        return None
    
    checkpoint_files = list(ckpt_path.glob("model_epoch*.pt"))
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the highest one
    latest_epoch = -1
    latest_file = None
    for ckpt_file in checkpoint_files:
        # Extract epoch number from filename like "model_epoch010.pt"
        match = re.search(r'model_epoch(\d+)\.pt$', ckpt_file.name)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_file = ckpt_file
    
    return latest_file

def load_checkpoint(ckpt_path: Path, model, optimizer, ema, device, is_ddp: bool):
    """Load checkpoint and return the epoch to resume from, plus the cached vocab."""
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Load model state
    model_to_load = model.module if is_ddp else model
    model_to_load.load_state_dict(checkpoint['model_state'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    # Load EMA state if available
    if ema and 'ema_state' in checkpoint:
        ema.shadow = checkpoint['ema_state']
        print("EMA state loaded from checkpoint.")
    elif ema:
        print("Warning: EMA enabled but no EMA state found in checkpoint.")
    
    # Get cached vocab to ensure consistency
    cached_vocab = checkpoint.get('vocab', None)
    if cached_vocab:
        print("Using cached vocabulary from checkpoint to ensure consistency.")
    
    resume_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {resume_epoch}")
    return resume_epoch, cached_vocab

# 6‑G. Trainer with self‑conditioning ----------------------------------------
def train(model, criterion, opt, epochs, steps_ep, loader, acp, ema, device, amp, sampler_cfg, is_ddp, ckpt_dir: str, args: argparse.Namespace):

    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    data_iter = iter(loader)
    rank = dist.get_rank() if is_ddp else 0
    ckpt_path = Path(ckpt_dir); ckpt_path.mkdir(exist_ok=True)
    pad_id = loader.dataset.vocab.get_stoi()["<pad>"]
    grad_clip = args.grad_clip
    grad_accum = args.grad_accum

    # Handle checkpoint resuming
    start_epoch = 1
    if args.resume and rank == 0:  # Only main process checks for checkpoints
        latest_ckpt = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            start_epoch, _ = load_checkpoint(latest_ckpt, model, opt, ema, device, is_ddp)
        else:
            print("No checkpoint found for resuming. Starting from epoch 1.")
    
    # Broadcast start_epoch to all processes in DDP
    if is_ddp:
        start_epoch_tensor = torch.tensor(start_epoch, device=device)
        dist.broadcast(start_epoch_tensor, 0)
        start_epoch = start_epoch_tensor.item()

    for ep in range(start_epoch, epochs + 1):
        model.train()
        running = 0
        tic = time.time()

        for step in range(steps_ep):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            x_tokens = batch.to(device)
            pad_mask = (x_tokens == pad_id)
            B = x_tokens.size(0)
            t = torch.randint(0, num_timesteps, (B,), device=device)

            # Get the underlying model when using DDP
            underlying_model = model.module if is_ddp else model
            clean = underlying_model.embedding(x_tokens)
            
            sq_a = get_schedule_values(acp.sqrt(), t, clean.shape)
            sq_1ma = get_schedule_values((1 - acp).sqrt(), t, clean.shape)
            noise = torch.randn_like(clean)
            noisy = sq_a * clean + sq_1ma * noise

            use_sc = random.random() < 0.5
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                # first pass (to obtain x0_pred)
                eps1, x0_1 = model(noisy, t, pad_mask, self_cond=None)
                # optional second pass with self‑cond
                eps = eps1
                if use_sc:
                    eps, _ = model(noisy, t, pad_mask, self_cond=x0_1.detach())

                loss = criterion(eps, noise) / grad_accum

            scaler.scale(loss).backward()
            running += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if ema: ema.update(model)

        if rank == 0:
            print(f"Epoch {ep:03}/{epochs:03}  loss={running/steps_ep:.4f}  {time.time()-tic:.1f}s")

            if ep % sampler_cfg["every"] == 0 or ep == epochs:
                # --- Save Checkpoint ---
                mdl = model.module if is_ddp else model
                ckpt_file = ckpt_path / f"model_epoch{ep:03}.pt"
                # Prepare checkpoint data
                checkpoint_data = {
                    "epoch": ep,
                    "model_state": mdl.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "vocab": loader.dataset.vocab,  # Cache vocab to avoid rebuilding
                    "args": args
                }
                
                # Add EMA state if available
                if ema:
                    checkpoint_data["ema_state"] = ema.shadow
                
                torch.save(checkpoint_data, ckpt_file)
                print(f" ↳ checkpoint saved: {ckpt_file}")

                # --- Generate Sample ---
                if ema: ema.apply_to(mdl)
                sampler = (DDIM_Sampler if sampler_cfg["kind"] == "ddim" else DDPM_Sampler)(
                    mdl, acp, num_timesteps, device,
                    **sampler_cfg.get("extra", {})
                )
                sample = sampler.sample(1, sequence_length)[0]
                print(" ↳ sample:", " ".join(loader.dataset.vocab.lookup_tokens(sample.tolist())))
                if ema: ema.restore(mdl)

# 6‑H. Datasets ---------------------------------------------------------------
class CodeDataset(Dataset):
    exts = {".py", ".js", ".ts", ".go", ".java", ".cs", ".cpp", ".c"}
    def __init__(self, root, tok, L):
        self.tok, self.L = tok, L
        # Store file paths and chunk indices instead of loading all content
        self.file_chunks = []  # [(file_path, start_idx, end_idx), ...]
        self._cache = {}  # Simple LRU-style cache for tokenized files
        self._cache_size = 100  # Cache up to 100 files
        
        for p in Path(root).rglob("*"):
            if p.suffix.lower() in self.exts:
                try:
                    text = p.read_text(encoding="utf8", errors="ignore")
                    toks = tok(text)
                    # Store chunk boundaries instead of actual chunks
                    for i in range(0, len(toks), L):
                        if i + L <= len(toks) or len(toks) - i >= L // 2:  # Keep partial chunks if >= half length
                            self.file_chunks.append((str(p), i, min(i + L, len(toks))))
                except Exception:
                    continue
        if not self.file_chunks:
            raise RuntimeError("no code found")

    def __len__(self): 
        return len(self.file_chunks)

    def __getitem__(self, i): 
        file_path, start_idx, end_idx = self.file_chunks[i]
        
        # Check cache first
        if file_path in self._cache:
            tokens = self._cache[file_path]
        else:
            # Load and tokenize file on demand
            try:
                text = Path(file_path).read_text(encoding="utf8", errors="ignore")
                tokens = self.tok(text)
                # Simple cache management - remove oldest if cache full
                if len(self._cache) >= self._cache_size:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[file_path] = tokens
            except Exception:
                return []  # Return empty if file can't be read
        
        return tokens[start_idx:end_idx]

def make_loader(batch_size: int,
                seq_len: int,
                data_dir: Optional[str] = None,
                is_ddp: bool = False,
                cached_vocab: Optional = None
                ) -> Tuple[DataLoader, Vocab, int]:
    """
    Creates a DataLoader for either the IMDB dataset or a custom CodeDataset.
    The collate function handles tokenization, numericalization, and padding.
    """
    tokenizer = get_tokenizer("basic_english")
    specials = ["<unk>", "<pad>"]
    pad_token = "<pad>"

    if data_dir:
        # --- Custom Code Dataset ---
        ds = CodeDataset(data_dir, tokenizer, seq_len)
        
        # Use cached vocab if provided (from checkpoint), otherwise build fresh
        if cached_vocab:
            print("Using provided cached vocabulary (skipping rebuild)")
            vocab = cached_vocab
        else:
            print("Building vocabulary from dataset...")
            # Build vocab from all tokens across all file chunks
            def token_generator():
                for i in range(len(ds)):
                    yield ds[i]
            vocab = build_vocab_from_iterator(token_generator(), specials=specials)
            vocab.set_default_index(vocab["<unk>"])
        
        pad_id = vocab[pad_token]

        def collate_fn(b: List[List[str]]):
            processed_batch = []
            for tokens in b:
                ids = vocab(tokens)
                if len(ids) < seq_len:
                    ids += [pad_id] * (seq_len - len(ids))
                else:
                    ids = ids[:seq_len]
                processed_batch.append(torch.tensor(ids, dtype=torch.long))
            return torch.stack(processed_batch)

    else:
        # --- IMDB Dataset ---
        imdb_data = [(label, text) for label, text in IMDB(split="train") if text.strip()]

        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)

        vocab = build_vocab_from_iterator(yield_tokens(imdb_data), specials=specials)
        vocab.set_default_index(vocab["<unk>"])
        pad_id = vocab[pad_token]
        ds = imdb_data

        def collate_fn(b: List[Tuple[int, str]]):
            processed_batch = []
            for _, text in b:
                ids = vocab(tokenizer(text))
                if len(ids) < seq_len:
                    ids += [pad_id] * (seq_len - len(ids))
                else:
                    ids = ids[:seq_len]
                processed_batch.append(torch.tensor(ids, dtype=torch.long))
            return torch.stack(processed_batch)

    sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_fn,
        drop_last=True,
    )
    # Attach vocab to the dataset for easy access during sampling
    loader.dataset.vocab = vocab
    return loader, vocab, pad_id

# 6‑I. CLI / entry‑point ------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--steps", type=int, default=steps_per_epoch)
    p.add_argument("--batch_size", type=int, default=batch_size)
    p.add_argument("--sequence_length", type=int, default=sequence_length)
    p.add_argument("--learning_rate", type=float, default=learning_rate)
    p.add_argument("--schedule", choices=["linear","cosine"], default="linear")
    p.add_argument("--sampler", choices=["ddpm","ddim"], default="ddpm")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--ddim-eta", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")

    p.add_argument("--phase", choices=["train", "generate"], default="train")
    args = p.parse_args()

    # DDP setup ---------------------------------------------------------------
    is_ddp = "WORLD_SIZE" in os.environ
    if is_ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}"); torch.cuda.set_device(device)
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Debug: Check DDP status
    print(f"DDP setup: is_ddp={is_ddp}, local_rank={local_rank}, resume={args.resume}")
    
    # Check for resume first to get correct vocab size ------------
    resume_vocab = None
    if args.resume:
        latest_ckpt = find_latest_checkpoint(args.ckpt_dir)
        if latest_ckpt:
            print(f"Found checkpoint for resuming: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            if 'vocab' in checkpoint:
                resume_vocab = checkpoint['vocab']
                print(f"Loaded vocab from checkpoint (size: {len(resume_vocab)})")
            else:
                print("No vocab found in checkpoint!")

    # data --------------------------------------------------------------------
    print(f"Building data loader...")
    # Pass cached vocab to make_loader to avoid rebuilding when resuming
    cached_vocab_to_use = resume_vocab if (args.resume and resume_vocab) else None
    loader, vocab, pad_id = make_loader(args.batch_size, args.sequence_length, args.data_dir, is_ddp, cached_vocab_to_use)
    print(f"Final vocab size: {len(vocab)}")
    
    vsize = len(vocab)

    # model -------------------------------------------------------------------
    model = DiffusionTransformerPE(vsize, hidden_size, num_layers, num_heads,
                                   dim_feedforward, args.sequence_length, pad_id).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    # This is the crucial change: EMA is initialized *after* DDP wrapping
    ema = EMA(model) if args.use_ema else None
    if local_rank == 0 and ema:
        print("EMA initialized.")

    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)
    acp = get_noise_schedule_v2(num_timesteps, args.schedule, device)

    sampler_cfg = {"kind": args.sampler,
                   "every": 10,
                   "extra": {"steps": args.ddim_steps, "eta": args.ddim_eta}}

    if args.phase == "train":
        train(model, criterion, opt, args.epochs, args.steps,
              loader, acp, ema, device, args.amp, sampler_cfg, is_ddp,
              args.ckpt_dir, args)
    elif args.phase == "generate":
        # Placeholder for generation logic
        pass

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
