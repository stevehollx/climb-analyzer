"""
File splitter utility for large files exceeding GitHub's 2GB release asset limit.

Splits SQLite databases >= 1.95GB into chunks < 1.9GB each, with SHA256 checksums
for verification.
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# GitHub release asset limit is 2GB, use 1.9GB for safety margin
DEFAULT_CHUNK_SIZE = 1_900_000_000  # 1.9 GB
SPLIT_THRESHOLD = 1_950_000_000     # 1.95 GB - when to trigger splitting


def should_split_file(file_path: Path) -> bool:
    """Check if file exceeds the split threshold (1.95GB)."""
    return file_path.stat().st_size >= SPLIT_THRESHOLD


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file using streaming."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def split_file(
    file_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    delete_original: bool = True
) -> Tuple[List[Path], Dict[str, str]]:
    """
    Split a large file into chunks with SHA256 checksums.

    Args:
        file_path: Path to the file to split
        chunk_size: Maximum size of each chunk in bytes (default 1.9GB)
        delete_original: Whether to delete the original file after splitting

    Returns:
        Tuple of (list of chunk paths, dict of checksums keyed by filename)
    """
    file_size = file_path.stat().st_size

    if file_size <= chunk_size:
        # No split needed - calculate checksum of original
        checksum = calculate_sha256(file_path)
        return [file_path], {file_path.name: checksum}

    chunks = []
    checksums = {}

    file_size_gb = file_size / (1024**3)
    logger.info(f"Splitting {file_path.name} ({file_size_gb:.2f} GB) into chunks...")
    print(f"  Splitting {file_path.name} ({file_size_gb:.2f} GB) into chunks...")

    with open(file_path, 'rb') as f:
        part_num = 1
        while True:
            chunk_data = f.read(chunk_size)
            if not chunk_data:
                break

            # Name chunks as .sqlite.001, .sqlite.002, etc.
            chunk_name = file_path.parent / f"{file_path.name}.{part_num:03d}"

            with open(chunk_name, 'wb') as out:
                out.write(chunk_data)

            # Calculate checksum for this chunk
            chunk_checksum = hashlib.sha256(chunk_data).hexdigest()
            checksums[chunk_name.name] = chunk_checksum
            chunks.append(chunk_name)

            chunk_size_gb = len(chunk_data) / (1024**3)
            logger.info(f"  Created {chunk_name.name} ({chunk_size_gb:.2f} GB)")
            print(f"    Created {chunk_name.name} ({chunk_size_gb:.2f} GB)")

            part_num += 1

    # Write checksums file in standard sha256sum format
    checksum_file = file_path.parent / f"{file_path.name}.sha256"
    with open(checksum_file, 'w') as f:
        for name, checksum in sorted(checksums.items()):
            # Standard format: "sha256hash  filename" (two spaces)
            f.write(f"{checksum}  {name}\n")

    logger.info(f"  Created checksum file: {checksum_file.name}")
    print(f"    Created checksum file: {checksum_file.name}")

    # Delete original if requested
    if delete_original and len(chunks) > 1:
        file_path.unlink()
        logger.info(f"  Deleted original file: {file_path.name}")
        print(f"    Deleted original file to save disk space")

    return chunks, checksums


def get_checksum_file_path(sqlite_file: Path) -> Optional[Path]:
    """Get the path to the checksum file for a split SQLite database."""
    checksum_path = sqlite_file.parent / f"{sqlite_file.name}.sha256"
    if checksum_path.exists():
        return checksum_path
    return None


def reassemble_file(chunk_paths: List[Path], output_path: Path) -> bool:
    """
    Reassemble a split file from its chunks.

    Args:
        chunk_paths: List of chunk paths in order
        output_path: Path for the reassembled file

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'wb') as out:
            for chunk_path in sorted(chunk_paths):
                with open(chunk_path, 'rb') as chunk:
                    while True:
                        data = chunk.read(8192)
                        if not data:
                            break
                        out.write(data)
        return True
    except Exception as e:
        logger.error(f"Failed to reassemble file: {e}")
        return False


def verify_chunks(chunk_paths: List[Path], checksums: Dict[str, str]) -> bool:
    """
    Verify chunk files against their checksums.

    Args:
        chunk_paths: List of chunk file paths
        checksums: Dict mapping filename to expected SHA256 checksum

    Returns:
        True if all chunks verify, False otherwise
    """
    for chunk_path in chunk_paths:
        expected = checksums.get(chunk_path.name)
        if not expected:
            logger.error(f"No checksum found for {chunk_path.name}")
            return False

        actual = calculate_sha256(chunk_path)
        if actual != expected:
            logger.error(f"Checksum mismatch for {chunk_path.name}")
            logger.error(f"  Expected: {expected}")
            logger.error(f"  Actual:   {actual}")
            return False

    return True
