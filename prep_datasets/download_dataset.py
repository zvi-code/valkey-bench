#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024-present, Zvi Schneider
#
# Flexible dataset downloader for vectordb-bench
#
# This file is part of valkey-search-benchmark and is licensed under the
# BSD 3-Clause License. See the LICENSE file in the root directory.
#
"""
Flexible dataset downloader for vectordb-bench
Downloads directly to NVMe at /mnt/data/datasets/

Usage examples:
  python download_any_dataset.py COHERE 1000000
  python download_any_dataset.py SIFT 5000000
  python download_any_dataset.py LAION 100000000
"""

import sys
import os
from vectordb_bench import config
from vectordb_bench.backend.dataset import Dataset

# Override the default dataset directory
# Use DATASET_PATH env var, or default to datasets/raw relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "datasets", "raw")
config.DATASET_LOCAL_DIR = os.environ.get("DATASET_PATH", DEFAULT_RAW_DIR)

def download_dataset(dataset_name, size):
    """Download a specific dataset with given size"""
    
    print("=" * 80)
    print(f"Downloading {dataset_name} dataset with {size:,} vectors")
    print("=" * 80)
    print(f"Target directory: {config.DATASET_LOCAL_DIR}")
    print()
    
    try:
        # Get the dataset enum
        dataset = Dataset[dataset_name.upper()]
        
        # Create manager with specified size
        manager = dataset.manager(size=size)
        
        # Print dataset info
        print(f"Dataset: {manager.data.name}")
        print(f"Size: {manager.data.size:,} vectors")
        print(f"Dimensions: {manager.data.dim}")
        print(f"Metric: {manager.data.metric_type}")
        print(f"\nStarting download...")
        print()
        
        # Download the dataset
        manager.prepare()
        
        print()
        print("=" * 80)
        print("✓ Download complete!")
        print("=" * 80)
        print(f"Location: {config.DATASET_LOCAL_DIR}/{manager.data.name.lower()}/")
        
        # Show disk usage
        os.system(f"df -h /mnt/data")
        
    except KeyError:
        print(f"Error: Unknown dataset '{dataset_name}'")
        print(f"Available datasets: {', '.join([d.name for d in Dataset])}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python download_any_dataset.py <DATASET_NAME> <SIZE>")
        print()
        print("Available datasets and sizes:")
        print("  LAION:  100000000")
        print("  GIST:   100000, 1000000")
        print("  COHERE: 100000, 1000000, 10000000")
        print("  BIOASQ: 1000000, 10000000")
        print("  GLOVE:  1000000")
        print("  SIFT:   500000, 5000000")
        print("  OPENAI: 500000, 5000000")
        print()
        print("Examples:")
        print("  python download_any_dataset.py COHERE 1000000")
        print("  python download_any_dataset.py SIFT 5000000")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    size = int(sys.argv[2])
    
    download_dataset(dataset_name, size)
