#!/usr/bin/env python3
"""
Quick test to debug vocab size mismatch without running full training
"""

import torch
from pathlib import Path
import sys

# Add current directory to path to import main.py functions
sys.path.append('.')
from main import find_latest_checkpoint, make_loader

def test_vocab_consistency():
    """Test vocab consistency between checkpoint and fresh build"""
    
    # Find latest checkpoint
    ckpt_dir = "checkpoints"
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    
    if not latest_ckpt:
        print("No checkpoint found!")
        return
        
    print(f"Found checkpoint: {latest_ckpt}")
    
    # Load checkpoint vocab
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(latest_ckpt, map_location=device)
    
    if 'vocab' in checkpoint:
        ckpt_vocab = checkpoint['vocab']
        print(f"Checkpoint vocab size: {len(ckpt_vocab)}")
        
        # Show some sample tokens
        print("Sample checkpoint vocab tokens:")
        sample_tokens = list(ckpt_vocab.get_stoi().keys())[:10]
        for token in sample_tokens:
            print(f"  '{token}' -> {ckpt_vocab[token]}")
    else:
        print("No vocab in checkpoint!")
        return
    
    # Build fresh vocab (no cached vocab)
    print("\nBuilding fresh vocab...")
    data_dir = "/home/theapemachine/go/src/github.com/theapemachine"
    loader, fresh_vocab, pad_id = make_loader(batch_size=1, seq_len=512, data_dir=data_dir, is_ddp=False, cached_vocab=None)
    print(f"Fresh vocab size: {len(fresh_vocab)}")
    
    # Test with cached vocab (simulate resume)
    print("\nTesting with cached vocab (simulating resume)...")
    loader_cached, cached_vocab_result, pad_id_cached = make_loader(batch_size=1, seq_len=512, data_dir=data_dir, is_ddp=False, cached_vocab=ckpt_vocab)
    print(f"Cached vocab result size: {len(cached_vocab_result)}")
    
    # Show some sample tokens
    print("Sample fresh vocab tokens:")
    sample_tokens = list(fresh_vocab.get_stoi().keys())[:10]
    for token in sample_tokens:
        print(f"  '{token}' -> {fresh_vocab[token]}")
    
    # Compare fresh vs checkpoint
    size_diff = len(fresh_vocab) - len(ckpt_vocab)
    print(f"\nSize difference (fresh vs checkpoint): {size_diff} tokens")
    
    # Check if cached vocab matches checkpoint perfectly
    print(f"Cached vocab matches checkpoint: {len(cached_vocab_result) == len(ckpt_vocab)}")
    
    if size_diff != 0:
        print("VOCAB MISMATCH DETECTED!")
        
        # Find what tokens are different
        ckpt_tokens = set(ckpt_vocab.get_stoi().keys())
        fresh_tokens = set(fresh_vocab.get_stoi().keys())
        
        only_in_fresh = fresh_tokens - ckpt_tokens
        only_in_ckpt = ckpt_tokens - fresh_tokens
        
        print(f"Tokens only in fresh vocab ({len(only_in_fresh)}):")
        for token in sorted(only_in_fresh)[:20]:  # Show first 20
            print(f"  '{token}'")
            
        print(f"Tokens only in checkpoint vocab ({len(only_in_ckpt)}):")
        for token in sorted(only_in_ckpt)[:20]:  # Show first 20
            print(f"  '{token}'")
    else:
        print("Vocab sizes match!")

if __name__ == "__main__":
    test_vocab_consistency()