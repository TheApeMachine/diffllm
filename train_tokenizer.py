
import os
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder

def train_bpe_tokenizer(data_dir: str, vocab_size: int, output_file: str):
    """
    Trains a BPE tokenizer on source code files and saves it, ignoring UTF-8 errors.
    """
    # 1. Find all source code files
    exts = {".py", ".js", ".ts", ".go", ".java", ".cs", ".cpp", ".c"}
    files = [p for p in Path(data_dir).rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise ValueError(f"No source code files found in {data_dir}")

    # 2. Define a generator to read files with error handling
    def file_iterator():
        for file_path in files:
            try:
                yield file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                # Silently skip files that can't be read
                continue

    # 3. Initialize a tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Metaspace()
    tokenizer.decoder = MetaspaceDecoder()

    # 4. Customize the trainer
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<pad>", "<s>", "</s>"])

    # 5. Train the tokenizer from the iterator
    print(f"Training BPE tokenizer on {len(files)} files...")
    tokenizer.train_from_iterator(file_iterator(), trainer=trainer, length=len(files))

    # 6. Save the tokenizer
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"Tokenizer trained and saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer for source code.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing source code files.")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Size of the vocabulary to build.")
    parser.add_argument("--output_file", type=str, default="checkpoints/bpe_tokenizer.json", help="Path to save the trained tokenizer.")
    args = parser.parse_args()

    train_bpe_tokenizer(args.data_dir, args.vocab_size, args.output_file)
