
import argparse
from pathlib import Path
from tokenizers import Tokenizer

def test_tokenizer(tokenizer_path: str, test_string: str):
    """
    Loads a tokenizer, encodes a string, decodes it, and verifies the reconstruction.
    """
    print(f"--- Tokenizer Verification ---")
    
    # 1. Load the tokenizer
    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer file not found at: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(path))
    print(f"Successfully loaded tokenizer from: {tokenizer_path}")
    
    # 2. Encode the test string
    print(f"\nOriginal string:\n---\n{test_string}\n---")
    encoded = tokenizer.encode(test_string)
    print(f"Encoded token IDs: {encoded.ids}")
    print(f"Encoded tokens: {encoded.tokens}")
    
    # 3. Decode the tokens
    decoded_string = tokenizer.decode(encoded.ids)
    print(f"\nDecoded string:\n---\n{decoded_string}\n---")
    
    # 4. Verify the reconstruction
    if test_string == decoded_string:
        print("\n✅ Verification SUCCESS: The decoded string perfectly matches the original.")
    else:
        print("\n❌ Verification FAILED: The decoded string does not match the original.")
        print(f"Original length: {len(test_string)}, Decoded length: {len(decoded_string)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained BPE tokenizer.")
    parser.add_argument(
        "--tokenizer-file", 
        type=str, 
        default="checkpoints/bpe_tokenizer.json", 
        help="Path to the trained tokenizer file."
    )
    parser.add_argument(
        "--test-string",
        type=str,
        default="def complex_function(x, y):\n    # This is a test\n    result = (x ** 2) + (y ** 2)\n    return result",
        help="The string to test encoding/decoding with."
    )
    args = parser.parse_args()

    test_tokenizer(args.tokenizer_file, args.test_string)
