
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tokenizers import Tokenizer

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

# --- 2. Hyper‑parameters (defaults) ---
# These can now be overridden by CLI args
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
    * self‑conditioning and prompt-conditioning (CFG) ready
    * returns (noise_pred, x0_pred, logits)
    """
    def __init__(self, vocab_size: int, hidden: int, layers: int,
                 heads: int, ff: int, max_len: int, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_id)
        self.pos_embedding = PositionalEncoding(max_len, hidden)
        self.time_embedding = TimestepEmbedding(hidden)
        self.sc_proj = nn.Linear(hidden, hidden, bias=False)
        self.prompt_proj = nn.Linear(hidden, hidden, bias=False)

        enc_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Linear(hidden, hidden * 2)

    def forward(self, noisy_emb: torch.Tensor, t: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None,
                self_cond: Optional[torch.Tensor] = None,
                prompt_emb: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self_cond is None: self_cond = torch.zeros_like(noisy_emb)
        if prompt_emb is None: prompt_emb = torch.zeros_like(noisy_emb)
        
        x = noisy_emb + self.sc_proj(self_cond) + self.prompt_proj(prompt_emb)
        x = x + self.pos_embedding(x) + self.time_embedding(t).unsqueeze(1)
        
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        noise_pred, x0_pred = self.head(x).chunk(2, dim=-1)
        
        logits = torch.matmul(x0_pred, self.embedding.weight.t())
        
        return noise_pred, x0_pred, logits

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

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        model_to_register = model.module if hasattr(model, 'module') else model
        for n, p in model_to_register.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self, model):
        model_to_update = model.module if hasattr(model, 'module') else model
        for n, p in model_to_update.named_parameters():
            if p.requires_grad:
                assert n in self.shadow
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_to(self, model: nn.Module):
        self.backup = {}
        model_to_apply = model.module if hasattr(model, 'module') else model
        for n, p in model_to_apply.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data
                p.data = self.shadow[n]

    def restore(self, model: nn.Module):
        model_to_restore = model.module if hasattr(model, 'module') else model
        for n, p in model_to_restore.named_parameters():
            if p.requires_grad and n in self.backup:
                p.data = self.backup[n]

def get_schedule_values(acp: torch.Tensor, t: torch.Tensor, shape):
    B = t.shape[0]
    al_t = acp.gather(0, t)
    return al_t.view(B, *((1,) * (len(shape) - 1))).expand(shape)

def embeddings_to_tokens(emb: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    logits = torch.matmul(emb, W.t())
    return F.gumbel_softmax(logits, tau=0.1, hard=True, dim=-1).argmax(-1)

class DDPM_Sampler(nn.Module):
    def __init__(self, model, acp, T, device, hidden_size):
        super().__init__()
        self.model, self.acp, self.T, self.device, self.hidden_size = model, acp, T, device, hidden_size
        acp_prev = torch.cat([torch.tensor([1.], device=device), acp[:-1]])
        self.alphas, self.betas = acp / acp_prev, 1 - acp / acp_prev
        self.sqrt_recip = (1 / self.alphas).sqrt()
        self.sqrt_om_acp = (1 - acp).sqrt()

    @torch.no_grad()
    def sample(self, B, L, pad_mask=None, prompt_emb=None, guidance_scale=7.5):
        x = torch.randn(B, L, self.hidden_size, device=self.device)
        self_cond = None
        for t_ind in reversed(range(self.T)):
            t = torch.full((B,), t_ind, dtype=torch.long, device=self.device)
            
            eps_cond, x0_cond, _ = self.model(x, t, pad_mask, self_cond, prompt_emb)
            eps_uncond, _, _ = self.model(x, t, pad_mask, self_cond, None)
            
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            self_cond = x0_cond

            mean = self.sqrt_recip[t_ind] * (x - self.betas[t_ind] / self.sqrt_om_acp[t_ind] * eps)
            if t_ind > 0:
                noise = torch.randn_like(x)
                x = mean + self.betas[t_ind].sqrt() * noise
            else:
                x = mean
        return embeddings_to_tokens(x, self.model.embedding.weight)

class DDIM_Sampler(nn.Module):
    def __init__(self, model, acp, T, device, hidden_size, steps=50, eta=0.0):
        super().__init__()
        self.model, self.device, self.eta, self.hidden_size = model, device, eta, hidden_size
        self.acp = acp
        self.steps = torch.linspace(0, T - 1, steps, dtype=torch.long)

    @torch.no_grad()
    def sample(self, B, L, pad_mask=None, prompt_emb=None, guidance_scale=7.5):
        x = torch.randn(B, L, self.hidden_size, device=self.device)
        self_cond = None
        for i in reversed(range(1, len(self.steps))):
            t, t_prev = self.steps[i], self.steps[i - 1]
            a_t, a_prev = self.acp[t], self.acp[t_prev]
            t_long = torch.full((B,), t.item(), dtype=torch.long, device=self.device)

            eps_cond, x0_cond, _ = self.model(x, t_long, pad_mask, self_cond, prompt_emb)
            eps_uncond, _, _ = self.model(x, t_long, pad_mask, self_cond, None)
            
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            self_cond = x0_cond

            x0_pred = (x - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            sigma = self.eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
            dir_xt = (1 - a_prev - sigma ** 2).sqrt() * eps
            noise = torch.randn_like(x) if sigma.item() > 0 else 0
            x = a_prev.sqrt() * x0_pred + dir_xt + sigma * noise
        return embeddings_to_tokens(x, self.model.embedding.weight)

def find_latest_checkpoint(ckpt_dir: str) -> Optional[Path]:
    ckpt_path = Path(ckpt_dir)
    if not ckpt_path.exists(): return None
    files = list(ckpt_path.glob("model_epoch*.pt"))
    if not files: return None
    return max(files, key=lambda f: int(re.search(r'model_epoch(\d+)\.pt$', f.name).group(1)))

def load_checkpoint(ckpt_path: Path, model, optimizer, ema, device, is_ddp: bool):
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    (model.module if is_ddp else model).load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    if ema and 'ema_state' in ckpt:
        ema.shadow = ckpt['ema_state']
    resume_epoch = ckpt['epoch'] + 1
    tokenizer_path = ckpt.get('tokenizer_path')
    return resume_epoch, tokenizer_path

def train(model, opt, epochs, steps_ep, loader, acp, ema, device, amp, sampler_cfg, is_ddp, ckpt_dir: str, args: argparse.Namespace, tokenizer: Tokenizer, pad_id: int):
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    mse_loss_fn = nn.MSELoss()
    data_iter = iter(loader)
    rank = dist.get_rank() if is_ddp else 0
    ckpt_path = Path(ckpt_dir); ckpt_path.mkdir(exist_ok=True)
    
    start_epoch = 1
    if args.resume and rank == 0:
        latest_ckpt = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            start_epoch, _ = load_checkpoint(latest_ckpt, model, opt, ema, device, is_ddp)

    if is_ddp:
        start_epoch_tensor = torch.tensor(start_epoch, device=device, dtype=torch.int)
        dist.broadcast(start_epoch_tensor, 0)
        start_epoch = start_epoch_tensor.item()

    for ep in range(start_epoch, epochs + 1):
        model.train()
        total_loss, total_mse, total_ce = 0, 0, 0
        tic = time.time()
        if is_ddp: loader.sampler.set_epoch(ep)

        for step in range(steps_ep):
            try:
                x_tokens, prompt_tokens = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x_tokens, prompt_tokens = next(data_iter)

            x_tokens, prompt_tokens = x_tokens.to(device), prompt_tokens.to(device)
            pad_mask = (x_tokens == pad_id)
            B, L = x_tokens.size()
            t = torch.randint(0, num_timesteps, (B,), device=device)

            underlying_model = model.module if is_ddp else model
            clean = underlying_model.embedding(x_tokens)
            
            prompt_emb = underlying_model.embedding(prompt_tokens)
            prompt_emb = prompt_emb.mean(dim=1, keepdim=True).expand(-1, L, -1)
            
            if random.random() < 0.1:
                prompt_emb = None

            sq_a = get_schedule_values(acp.sqrt(), t, clean.shape)
            sq_1ma = get_schedule_values((1 - acp).sqrt(), t, clean.shape)
            noise = torch.randn_like(clean)
            noisy = sq_a * clean + sq_1ma * noise

            use_sc = random.random() < 0.5
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                eps1, x0_1, logits1 = model(noisy, t, pad_mask, None, prompt_emb)
                eps, logits = eps1, logits1
                if use_sc:
                    eps, _, logits = model(noisy, t, pad_mask, x0_1.detach(), prompt_emb)
                
                loss_mse = mse_loss_fn(eps, noise)
                loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), x_tokens.view(-1), ignore_index=pad_id)
                loss = (args.mse_lambda * loss_mse + (1 - args.mse_lambda) * loss_ce) / args.grad_accum

            scaler.scale(loss).backward()
            total_loss += loss.item() * args.grad_accum
            total_mse += loss_mse.item()
            total_ce += loss_ce.item()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if ema: ema.update(model)

        if rank == 0:
            avg_loss, avg_mse, avg_ce = total_loss/steps_ep, total_mse/steps_ep, total_ce/steps_ep
            print(f"Epoch {ep:03}/{epochs:03} Loss={avg_loss:.4f} (MSE:{avg_mse:.4f}, CE:{avg_ce:.4f}) {time.time()-tic:.1f}s")

            if ep % sampler_cfg["every"] == 0 or ep == epochs:
                mdl = model.module if is_ddp else model
                ckpt_file = ckpt_path / f"model_epoch{ep:03}.pt"
                torch.save({
                    "epoch": ep, "model_state": mdl.state_dict(),
                    "optimizer_state": opt.state_dict(), "tokenizer_path": args.tokenizer_file,
                    "args": args, **({"ema_state": ema.shadow} if ema else {})
                }, ckpt_file)
                print(f" ↳ checkpoint saved: {ckpt_file}")

                if ema: ema.apply_to(mdl)
                sampler = (DDIM_Sampler if sampler_cfg["kind"] == "ddim" else DDPM_Sampler)(
                    mdl, acp, num_timesteps, device, args.hidden_size, **sampler_cfg.get("extra", {})
                )
                sample = sampler.sample(1, args.sequence_length, prompt_emb=prompt_emb[:1] if prompt_emb is not None else None, guidance_scale=args.guidance_scale)[0]
                print(" ↳ sample:", tokenizer.decode(sample.tolist()))
                if ema: ema.restore(mdl)

class CodeDataset(Dataset):
    exts = {".py", ".js", ".ts", ".go", ".java", ".cs", ".cpp", ".c"}
    def __init__(self, root: str, tokenizer: Tokenizer, seq_len: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        # Store file paths and pre-calculated chunk boundaries, not tokenized data
        self.file_chunks = []
        self._cache = {} # Cache for recently tokenized files
        self._cache_size = 200 # Number of tokenized files to cache in RAM

        print("Scanning dataset...")
        for p in Path(root).rglob("*"):
            if p.suffix.lower() in self.exts:
                try:
                    # Quick scan to determine number of tokens and create chunk boundaries
                    text = p.read_text(encoding="utf-8", errors="ignore")
                    num_toks = len(self.tokenizer.encode(text).ids)
                    for i in range(0, num_toks, seq_len):
                        # Keep partial chunks if they are at least half length
                        if i + seq_len <= num_toks or num_toks - i >= seq_len // 2:
                            self.file_chunks.append((str(p), i, min(i + seq_len, num_toks)))
                except Exception:
                    continue
        if not self.file_chunks:
            raise RuntimeError(f"No valid code files found in {root}")
        print(f"Found {len(self.file_chunks)} chunks across {len(set(fc[0] for fc in self.file_chunks))} files.")

    def __len__(self): 
        return len(self.file_chunks)

    def _get_chunk_from_file(self, file_path_str: str, start_idx: int, end_idx: int) -> List[int]:
        """Helper to get a specific token chunk, using a cache."""
        if file_path_str in self._cache:
            all_ids = self._cache[file_path_str]
        else:
            try:
                text = Path(file_path_str).read_text(encoding="utf-8", errors="ignore")
                all_ids = self.tokenizer.encode(text).ids
                if len(self._cache) >= self._cache_size:
                    # Simple LRU-like eviction
                    del self._cache[next(iter(self._cache))]
                self._cache[file_path_str] = all_ids
            except Exception:
                return [] # Return empty on read error
        return all_ids[start_idx:end_idx]

    def __getitem__(self, i): 
        target_path, target_start, target_end = self.file_chunks[i]
        target_chunk = self._get_chunk_from_file(target_path, target_start, target_end)

        # For prompts, randomly select another chunk from the dataset
        rand_idx = random.randint(0, len(self.file_chunks) - 1)
        prompt_path, prompt_start, prompt_end = self.file_chunks[rand_idx]
        prompt_chunk = self._get_chunk_from_file(prompt_path, prompt_start, prompt_end)
        
        # Ensure chunks are not empty
        if not target_chunk or not prompt_chunk:
             # Fallback to a non-empty random chunk if something goes wrong
            return self.__getitem__(random.randint(0, len(self) - 1))

        return target_chunk, prompt_chunk

def make_loader(batch_size: int, seq_len: int, data_dir: str, tokenizer: Tokenizer, is_ddp: bool = False) -> Tuple[DataLoader, int]:
    ds = CodeDataset(data_dir, tokenizer, seq_len)
    pad_id = tokenizer.token_to_id("<pad>")
    if pad_id is None: raise ValueError("Tokenizer must have a <pad> token.")

    def collate_fn(b: List[Tuple[List[int], List[int]]]):
        targets, prompts = [], []
        for target_ids, prompt_ids in b:
            if not target_ids or not prompt_ids: continue # Skip if getitem failed
            # Pad target
            targets.append(torch.tensor((target_ids + [pad_id] * seq_len)[:seq_len], dtype=torch.long))
            # Pad prompt
            prompts.append(torch.tensor((prompt_ids + [pad_id] * seq_len)[:seq_len], dtype=torch.long))

        if not targets: return torch.empty(0), torch.empty(0)
        return torch.stack(targets), torch.stack(prompts)

    sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler, collate_fn=collate_fn, drop_last=True, num_workers=4, pin_memory=True)
    return loader, pad_id

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--steps", type=int, default=steps_per_epoch)
    p.add_argument("--batch_size", type=int, default=batch_size)
    p.add_argument("--learning_rate", type=float, default=learning_rate)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--use-ema", action="store_true")
    p.add_argument("--resume", action="store_true")
    
    p.add_argument("--sequence_length", type=int, default=sequence_length)
    p.add_argument("--hidden-size", type=int, default=hidden_size)
    p.add_argument("--num-layers", type=int, default=num_layers)
    p.add_argument("--num-heads", type=int, default=num_heads)
    p.add_argument("--dim-feedforward", type=int, default=dim_feedforward)

    p.add_argument("--schedule", choices=["linear","cosine"], default="linear")
    p.add_argument("--sampler", choices=["ddpm","ddim"], default="ddpm")
    p.add_argument("--ddim-steps", type=int, default=50)
    p.add_argument("--ddim-eta", type=float, default=0.0)
    p.add_argument("--mse-lambda", type=float, default=0.5)
    p.add_argument("--guidance-scale", type=float, default=7.5)

    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--tokenizer-file", type=str, default="checkpoints/bpe_tokenizer.json")
    
    p.add_argument("--phase", choices=["train", "generate"], default="train")
    p.add_argument("--prompt", type=str, default=None)
    args = p.parse_args()

    is_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}"); torch.cuda.set_device(device)
    else:
        local_rank, device = 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if local_rank == 0: print(f"DDP: {is_ddp}, Rank: {local_rank}, Device: {device}")

    tokenizer_path = args.tokenizer_file
    if args.resume and args.phase == "train":
        latest_ckpt = find_latest_checkpoint(args.ckpt_dir)
        if latest_ckpt:
            ckpt = torch.load(latest_ckpt, map_location='cpu')
            if ckpt.get('tokenizer_path'): tokenizer_path = ckpt['tokenizer_path']
    
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    
    tokenizer = Tokenizer.from_file(tokenizer_path)
    if local_rank == 0: print(f"Tokenizer loaded (vocab size: {tokenizer.get_vocab_size()})")
    
    data_dir = args.data_dir if args.data_dir else "."
    vsize, pad_id = tokenizer.get_vocab_size(), tokenizer.token_to_id("<pad>")

    model = DiffusionTransformerPE(vsize, args.hidden_size, args.num_layers, args.num_heads, args.dim_feedforward, args.sequence_length, pad_id).to(device)
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)

    if is_ddp: model = DDP(model, device_ids=[local_rank])
    ema = EMA(model) if args.use_ema else None
    acp = get_noise_schedule_v2(num_timesteps, args.schedule, device)
    sampler_cfg = {"kind": args.sampler, "every": 10, "extra": {"steps": args.ddim_steps, "eta": args.ddim_eta}}

    if args.phase == "train":
        loader, _ = make_loader(args.batch_size, args.sequence_length, data_dir, tokenizer, is_ddp)
        train(model, opt, args.epochs, args.steps, loader, acp, ema, device, args.amp, sampler_cfg, is_ddp, args.ckpt_dir, args, tokenizer, pad_id)
    elif args.phase == "generate":
        # Generation is handled by generate.py
        if local_rank == 0: print("To generate samples, please use generate.py")
        pass

    if "WORLD_SIZE" in os.environ: dist.destroy_process_group()

if __name__ == "__main__":
    main()
