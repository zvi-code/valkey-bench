#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024-present, Zvi Schneider
#
# Unified Dataset Manager for Valkey Vector Benchmarking
#
# This file is part of valkey-search-benchmark and is licensed under the
# BSD 3-Clause License. See the LICENSE file in the root directory.
#
"""
Unified Dataset Manager for Valkey Vector Benchmarking

Handles downloading, converting, and verifying vector datasets from multiple sources:
- ANN-Benchmarks (standard ML datasets)
- BigANN (billion-scale subsets)
- VectorDBBench (modern embedding datasets)
- YFCC-10M (metadata filtering)

Usage:
    python dataset_manager.py list [--filter <name>]
    python dataset_manager.py get <dataset-name> [--force]
    python dataset_manager.py verify <dataset.bin>
    python dataset_manager.py convert --input <file> --output <file> [options]
"""

import os
import sys
import json
import struct
import hashlib
import argparse
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Determine Python command to use (prefer venv if available)
VENV_PYTHON = Path(__file__).parent.parent / "venv" / "bin" / "python3"
PYTHON_CMD = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

# Paths - configurable via environment variables
# Falls back to local project directories if /mnt/data is not available
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

def _get_default_datasets_dir() -> Path:
    """Get default datasets directory, preferring /mnt/data if writable."""
    mnt_path = Path("/mnt/data/datasets")
    if mnt_path.parent.exists():
        try:
            mnt_path.mkdir(parents=True, exist_ok=True)
            return mnt_path
        except PermissionError:
            pass
    return PROJECT_ROOT / "datasets" / "raw"

def _get_default_build_dir() -> Path:
    """Get default build directory, preferring /mnt/data if writable."""
    mnt_path = Path("/mnt/data/build-datasets")
    if mnt_path.parent.exists():
        try:
            mnt_path.mkdir(parents=True, exist_ok=True)
            return mnt_path
        except PermissionError:
            pass
    return PROJECT_ROOT / "datasets"

DATASETS_DIR = Path(os.environ.get("DATASET_PATH", "")) if os.environ.get("DATASET_PATH") else _get_default_datasets_dir()
BUILD_DIR = Path(os.environ.get("BUILD_DATASET_PATH", "")) if os.environ.get("BUILD_DATASET_PATH") else _get_default_build_dir()
CONVERSION_DIR = PROJECT_ROOT / "prep_datasets"
UTILS_DIR = PROJECT_ROOT / "utils" / "datasets"

# Binary format constants
DATASET_MAGIC = 0xDECDB001
DATASET_VERSION_1 = 1
DATASET_VERSION_2 = 2

# Dataset catalog
DATASETS = {
    # Small datasets (< 100K vectors)
    "mnist": {
        "source": "ann-benchmarks",
        "url": "http://ann-benchmarks.com/mnist-784-euclidean.hdf5",
        "vectors": 60000,
        "dims": 784,
        "metric": "L2",
        "description": "MNIST digits (60K vectors, 784 dims)"
    },
    "fashion-mnist": {
        "source": "ann-benchmarks",
        "url": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "vectors": 60000,
        "dims": 784,
        "metric": "L2",
        "description": "Fashion MNIST (60K vectors, 784 dims)"
    },
    "cohere-small-100k": {
        "source": "vectordb-bench",
        "dataset_type": "COHERE",
        "size": 100000,
        "vectors": 100000,
        "dims": 768,
        "metric": "COSINE",
        "description": "Cohere embeddings (100K vectors, 768 dims)"
    },
    
    # Medium datasets (1M vectors)
    "sift-128": {
        "source": "ann-benchmarks",
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "vectors": 1000000,
        "dims": 128,
        "metric": "L2",
        "description": "SIFT descriptors (1M vectors, 128 dims)"
    },
    "gist-960": {
        "source": "ann-benchmarks",
        "url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "vectors": 1000000,
        "dims": 960,
        "metric": "L2",
        "description": "GIST descriptors (1M vectors, 960 dims)"
    },
    "glove-25": {
        "source": "ann-benchmarks",
        "url": "http://ann-benchmarks.com/glove-25-angular.hdf5",
        "vectors": 1183514,
        "dims": 25,
        "metric": "COSINE",
        "description": "GloVe word embeddings (1.18M vectors, 25 dims)"
    },
    "glove-50": {
        "source": "ann-benchmarks",
        "url": "http://ann-benchmarks.com/glove-50-angular.hdf5",
        "vectors": 1183514,
        "dims": 50,
        "metric": "COSINE",
        "description": "GloVe word embeddings (1.18M vectors, 50 dims)"
    },
    "glove-100": {
        "source": "ann-benchmarks",
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "vectors": 1183514,
        "dims": 100,
        "metric": "COSINE",
        "description": "GloVe word embeddings (1.18M vectors, 100 dims)"
    },
    "cohere-medium-1m": {
        "source": "vectordb-bench",
        "dataset_type": "COHERE",
        "size": 1000000,
        "vectors": 1000000,
        "dims": 768,
        "metric": "COSINE",
        "description": "Cohere embeddings (1M vectors, 768 dims)"
    },
    
    # Large datasets (5-10M vectors)
    "deep-96": {
        "source": "ann-benchmarks",
        "url": "http://ann-benchmarks.com/deep-image-96-angular.hdf5",
        "vectors": 10000000,
        "dims": 96,
        "metric": "COSINE",
        "description": "Deep image embeddings (10M vectors, 96 dims)"
    },
    "bigann-10m": {
        "source": "bigann",
        "base_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks",
        "vectors": 10000000,
        "dims": 128,
        "metric": "L2",
        "description": "BigANN SIFT subset (10M vectors, 128 dims)",
        "files": {
            "base": "bigann/base.10M.u8bin",
            "queries": "bigann/query.public.10K.u8bin",
            "gt": "GT_10M/bigann-10M"
        }
    },
    "deep-10m": {
        "source": "bigann",
        "base_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks",
        "vectors": 10000000,
        "dims": 256,
        "metric": "L2",
        "description": "Deep1B subset (10M vectors, 256 dims)",
        "files": {
            "base": "deep1b/base.10M.fbin",
            "queries": "deep1b/query.public.10K.fbin",
            "gt": "GT_10M/deep-10M"
        }
    },
    "yfcc-10m": {
        "source": "bigann-metadata",
        "base_url": "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M",
        "vectors": 10000000,
        "dims": 192,
        "metric": "L2",
        "has_metadata": True,
        "description": "YFCC-10M with metadata (10M vectors, 192 dims, 200K tags)",
        "files": {
            "base": "base.10M.u8bin",
            "queries": "query.public.100K.u8bin",
            "gt": "GT.public.100K.ibin",
            "base_metadata": "base.metadata.10M.spmat",
            "query_metadata": "query.metadata.public.100K.spmat",
            "vocabulary": "yfcc100M.vocab.words.txt"
        }
    },
    "cohere-large-10m": {
        "source": "vectordb-bench",
        "dataset_type": "COHERE",
        "size": 10000000,
        "vectors": 10000000,
        "dims": 768,
        "metric": "COSINE",
        "description": "Cohere embeddings (10M vectors, 768 dims)"
    },
    "openai-medium-500k": {
        "source": "vectordb-bench",
        "dataset_type": "OPENAI",
        "size": 500000,
        "vectors": 500000,
        "dims": 1536,
        "metric": "COSINE",
        "description": "OpenAI embeddings (500K vectors, 1536 dims)"
    },
    "openai-large-5m": {
        "source": "vectordb-bench",
        "dataset_type": "OPENAI",
        "size": 5000000,
        "vectors": 5000000,
        "dims": 1536,
        "metric": "COSINE",
        "description": "OpenAI embeddings (5M vectors, 1536 dims)"
    },
}


def list_datasets(filter_name: Optional[str] = None):
    """List all available datasets with optional filtering."""
    print("\n" + "="*80)
    print("Available Datasets")
    print("="*80 + "\n")
    
    # Group by size
    small = []
    medium = []
    large = []
    
    for name, info in sorted(DATASETS.items()):
        if filter_name and filter_name.lower() not in name.lower():
            continue
        
        vectors = info["vectors"]
        if vectors < 100000:
            small.append((name, info))
        elif vectors <= 1500000:
            medium.append((name, info))
        else:
            large.append((name, info))
    
    def print_group(title, datasets):
        if not datasets:
            return
        print(f"\n{title}:")
        print("-" * 80)
        for name, info in datasets:
            vectors_str = f"{info['vectors']:,}" if info['vectors'] < 1000000 else f"{info['vectors'] // 1000000}M"
            has_meta = " [+metadata]" if info.get("has_metadata") else ""
            print(f"  {name:25} {info['description']:45} {has_meta}")
        
    print_group("Small Datasets (< 100K vectors)", small)
    print_group("Medium Datasets (1M vectors)", medium)
    print_group("Large Datasets (5-10M vectors)", large)
    
    print("\n" + "="*80)
    print(f"Total: {len(small) + len(medium) + len(large)} datasets")
    print("="*80 + "\n")
    
    print("Usage:")
    print("  ./dataset.sh get <dataset-name>")
    print("  ./dataset.sh get mnist")
    print("  ./dataset.sh get cohere-medium-1m")
    print()


def download_file(url: str, output_path: Path, description: str = "file"):
    """Download a file with progress."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Output: {output_path}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ Downloaded successfully")
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def get_dataset(dataset_name: str, force: bool = False):
    """Download and convert a dataset."""
    if dataset_name not in DATASETS:
        print(f"✗ Unknown dataset: {dataset_name}")
        print(f"  Run './dataset.sh list' to see available datasets")
        return False
    
    info = DATASETS[dataset_name]
    output_bin = BUILD_DIR / f"{dataset_name}.bin"
    
    # Check if already exists
    if output_bin.exists() and not force:
        print(f"✓ Dataset already exists: {output_bin}")
        print(f"  Use --force to re-download")
        return True
    
    print(f"\n{'='*80}")
    print(f"Downloading and Converting: {dataset_name}")
    print(f"{'='*80}\n")
    print(f"Description: {info['description']}")
    print(f"Vectors: {info['vectors']:,}")
    print(f"Dimensions: {info['dims']}")
    print(f"Metric: {info['metric']}")
    print()
    
    # Create directories
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    source = info["source"]
    
    # Download and convert the dataset
    success = False
    if source == "ann-benchmarks":
        success = get_ann_benchmarks_dataset(dataset_name, info, output_bin)
    elif source == "vectordb-bench":
        success = get_vectordb_bench_dataset(dataset_name, info, output_bin)
    elif source == "bigann":
        success = get_bigann_dataset(dataset_name, info, output_bin)
    elif source == "bigann-metadata":
        success = get_bigann_metadata_dataset(dataset_name, info, output_bin)
    else:
        print(f"✗ Unsupported source: {source}")
        return False
    
    # Create symlink in local datasets directory if successful
    if success:
        create_dataset_symlink(dataset_name)
    
    return success


def get_ann_benchmarks_dataset(name: str, info: Dict, output_bin: Path) -> bool:
    """Download and convert ANN-Benchmarks HDF5 dataset."""
    hdf5_path = DATASETS_DIR / f"{name}.hdf5"
    
    # Download HDF5
    if not hdf5_path.exists():
        if not download_file(info["url"], hdf5_path, f"{name} HDF5"):
            return False
    else:
        print(f"✓ Using cached HDF5: {hdf5_path}")
    
    # Convert to binary using prepare_binary.py
    print(f"\nConverting to Valkey binary format...")
    prepare_binary = CONVERSION_DIR / "prepare_binary.py"
    
    cmd = [
        PYTHON_CMD, str(prepare_binary),
        str(hdf5_path),
        str(output_bin),
        "--metric", info["metric"],
        "--max-neighbors", "100"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"✓ Dataset ready: {output_bin}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed: {e}")
        print(e.stderr)
        return False


def get_vectordb_bench_dataset(name: str, info: Dict, output_bin: Path) -> bool:
    """Download and convert VectorDBBench dataset."""
    # Download parquet files
    download_script = CONVERSION_DIR / "download_dataset.py"
    parquet_dir = DATASETS_DIR / info["dataset_type"].lower() / f"{info['dataset_type'].lower()}_{name.replace('-', '_').replace(info['dataset_type'].lower() + '_', '')}"
    
    print(f"\nDownloading VectorDBBench dataset...")
    cmd = [
        PYTHON_CMD, str(download_script),
        info["dataset_type"],
        str(info["size"])
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e}")
        return False
    
    # Convert parquet to HDF5
    hdf5_path = DATASETS_DIR / f"{name}.hdf5"
    convert_script = CONVERSION_DIR / "convert_parquet_to_hdf5.py"
    
    print(f"\nConverting Parquet to HDF5...")
    cmd = [
        PYTHON_CMD, str(convert_script),
        str(parquet_dir),
        str(hdf5_path),
        "--name", name
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion to HDF5 failed: {e}")
        return False
    
    # Convert HDF5 to binary
    prepare_binary = CONVERSION_DIR / "prepare_binary.py"
    
    print(f"\nConverting to Valkey binary format...")
    cmd = [
        PYTHON_CMD, str(prepare_binary),
        str(hdf5_path),
        str(output_bin),
        "--metric", info["metric"],
        "--max-neighbors", "100"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Dataset ready: {output_bin}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Conversion failed: {e}")
        return False


def get_bigann_dataset(name: str, info: Dict, output_bin: Path) -> bool:
    """Download and convert BigANN dataset."""
    print(f"✗ BigANN dataset conversion not yet implemented")
    print(f"  Dataset: {name}")
    print(f"  Use ann-benchmarks or vectordb-bench datasets for now")
    return False


def get_bigann_metadata_dataset(name: str, info: Dict, output_bin: Path) -> bool:
    """Download and convert BigANN dataset with metadata (YFCC-10M)."""
    print(f"✗ BigANN metadata dataset conversion not yet implemented")
    print(f"  Dataset: {name}")
    print(f"  This requires special handling for metadata")
    return False


def create_dataset_symlink(dataset_name: str):
    """Create a symlink in the local datasets directory pointing to the build directory."""
    # Local datasets directory for symlinks
    local_datasets_dir = PROJECT_ROOT / "datasets"
    local_datasets_dir.mkdir(parents=True, exist_ok=True)
    
    # Source file in build directory
    source_bin = BUILD_DIR / f"{dataset_name}.bin"
    
    # Symlink in local datasets directory
    symlink_path = local_datasets_dir / f"{dataset_name}.bin"
    
    # Skip symlink if source and target are the same path
    if source_bin.resolve() == symlink_path.resolve() or source_bin == symlink_path:
        # File is already in the right place, no symlink needed
        return True
    
    # Remove existing symlink if it exists
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()
    
    # Create symlink
    try:
        symlink_path.symlink_to(source_bin)
        print(f"✓ Created symlink: {symlink_path} -> {source_bin}")
        return True
    except Exception as e:
        print(f"⚠ Warning: Could not create symlink: {e}")
        return False


def verify_dataset(bin_path: Path) -> bool:
    """Verify a binary dataset file."""
    if not bin_path.exists():
        print(f"✗ File not found: {bin_path}")
        return False
    
    try:
        with open(bin_path, 'rb') as f:
            # Read header
            magic = struct.unpack('<I', f.read(4))[0]
            version = struct.unpack('<I', f.read(4))[0]
            
            if magic != DATASET_MAGIC:
                print(f"✗ Invalid magic number: {hex(magic)} (expected {hex(DATASET_MAGIC)})")
                return False
            
            if version not in [DATASET_VERSION_1, DATASET_VERSION_2]:
                print(f"✗ Invalid version: {version}")
                return False
            
            num_vectors = struct.unpack('<Q', f.read(8))[0]
            num_queries = struct.unpack('<Q', f.read(8))[0]
            dim = struct.unpack('<I', f.read(4))[0]
            num_neighbors = struct.unpack('<I', f.read(4))[0]
            metric = struct.unpack('<B', f.read(1))[0]
            
            metric_names = {0: "L2", 1: "IP", 2: "COSINE"}
            metric_str = metric_names.get(metric, f"Unknown({metric})")
            
            print(f"✓ Valid dataset: {bin_path.name}")
            print(f"  Version: {version}")
            print(f"  Vectors: {num_vectors:,}")
            print(f"  Queries: {num_queries:,}")
            print(f"  Dimensions: {dim}")
            print(f"  Neighbors: {num_neighbors}")
            print(f"  Metric: {metric_str}")
            
            if version == DATASET_VERSION_2:
                # Read metadata flag
                f.seek(29)  # Skip to has_metadata field
                has_metadata = struct.unpack('<B', f.read(1))[0]
                if has_metadata:
                    vocab_size = struct.unpack('<I', f.read(4))[0]
                    print(f"  Metadata: Yes (vocabulary size: {vocab_size:,})")
            
            return True
            
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Unified Dataset Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List available datasets")
    list_parser.add_argument("--filter", help="Filter datasets by name")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Download and convert dataset")
    get_parser.add_argument("dataset", help="Dataset name")
    get_parser.add_argument("--force", action="store_true", help="Force re-download")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify binary dataset")
    verify_parser.add_argument("file", help="Path to .bin file")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert custom dataset")
    convert_parser.add_argument("--input", required=True, help="Input file (HDF5)")
    convert_parser.add_argument("--output", required=True, help="Output file (.bin)")
    convert_parser.add_argument("--metric", default="L2", help="Distance metric")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_datasets(args.filter)
    elif args.command == "get":
        success = get_dataset(args.dataset, args.force)
        sys.exit(0 if success else 1)
    elif args.command == "verify":
        success = verify_dataset(Path(args.file))
        sys.exit(0 if success else 1)
    elif args.command == "convert":
        # Call prepare_binary.py
        prepare_binary = CONVERSION_DIR / "prepare_binary.py"
        cmd = [
            PYTHON_CMD, str(prepare_binary),
            args.input,
            args.output,
            "--metric", args.metric,
            "--max-neighbors", "100"
        ]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
