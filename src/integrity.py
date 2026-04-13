"""
Deterministic SHA-256 hashing utilities for files, byte content, and strings.

All functions return hash strings prefixed with 'sha256:' for consistency.
Uses only Python standard library (no external dependencies).
"""

import hashlib
from pathlib import Path


def sha256_file(path: str) -> str:
    """
    Return sha256 hash of a file as 'sha256:<hex>'.
    
    Files are read in chunks to avoid loading entire file into memory.
    
    Args:
        path: Path to the file to hash
        
    Returns:
        Hash string in format 'sha256:<hex_digest>'
        
    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If the file cannot be read
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    sha256 = hashlib.sha256()
    
    # Read file in chunks to avoid loading entire file into memory
    with file_path.open('rb') as f:
        while chunk := f.read(8192):  # 8KB chunks
            sha256.update(chunk)
    
    return f"sha256:{sha256.hexdigest()}"


def sha256_bytes(data: bytes) -> str:
    """
    Return sha256 hash of bytes as 'sha256:<hex>'.
    
    Args:
        data: Bytes to hash
        
    Returns:
        Hash string in format 'sha256:<hex_digest>'
    """
    sha256 = hashlib.sha256(data)
    return f"sha256:{sha256.hexdigest()}"


def sha256_string(text: str) -> str:
    """
    Return sha256 hash of UTF-8 encoded string as 'sha256:<hex>'.
    
    Args:
        text: String to hash (will be UTF-8 encoded)
        
    Returns:
        Hash string in format 'sha256:<hex_digest>'
    """
    encoded = text.encode('utf-8')
    return sha256_bytes(encoded)


if __name__ == "__main__":
    """Self-check: print example hashes."""
    print("=== Integrity Module Self-Check ===\n")
    
    # Hash a string
    test_string = "prooflink"
    string_hash = sha256_string(test_string)
    print(f"String: '{test_string}'")
    print(f"Hash:   {string_hash}\n")
    
    # Hash bytes
    test_bytes = b"proof\x00link"
    bytes_hash = sha256_bytes(test_bytes)
    print(f"Bytes:  {test_bytes!r}")
    print(f"Hash:   {bytes_hash}\n")
    
    print("Self-check complete.")

