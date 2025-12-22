#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024-present, Zvi Schneider
#
# Script to investigate dataset structures and ground truth data
#
# This file is part of valkey-search-benchmark and is licensed under the
# BSD 3-Clause License. See the LICENSE file in the root directory.
#

"""
Script to investigate the actual structure of modern datasets
to see if they have proper ground truth that we missed
"""

import sys

def check_huggingface_dataset(dataset_name):
    """Check what's actually in a Hugging Face dataset"""
    try:
        from datasets import load_dataset
        print(f"Checking Hugging Face dataset: {dataset_name}")

        # Try to load a small sample first
        dataset = load_dataset(dataset_name, split="train", streaming=True)

        # Get first few items to see structure
        first_items = []
        for i, item in enumerate(dataset):
            first_items.append(item)
            if i >= 3:  # Just get first 3 items
                break

        print(f"Dataset structure:")
        for i, item in enumerate(first_items):
            print(f"  Item {i} keys: {list(item.keys())}")
            for key, value in item.items():
                if hasattr(value, '__len__') and not isinstance(value, str):
                    print(f"    {key}: type={type(value)}, length={len(value)}")
                else:
                    print(f"    {key}: type={type(value)}, value={str(value)[:100]}...")

        return True

    except Exception as e:
        print(f"Error checking {dataset_name}: {e}")
        return False

def check_beir_dataset(dataset_name):
    """Check what's available in BEIR datasets"""
    try:
        print(f"Checking BEIR dataset: {dataset_name}")

        # Try to install BEIR if not available
        try:
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader
        except ImportError:
            print("Installing BEIR...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "beir"])
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader

        # Download a small dataset
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = util.download_and_unzip(url, "/tmp/beir_check")

        # Load the data
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        print(f"BEIR {dataset_name} structure:")
        print(f"  Corpus: {len(corpus)} documents")
        print(f"  Queries: {len(queries)} queries")
        print(f"  Qrels: {len(qrels)} query-document pairs")

        # Show sample data
        sample_corpus_id = list(corpus.keys())[0]
        sample_query_id = list(queries.keys())[0]

        print(f"  Sample corpus item: {corpus[sample_corpus_id]}")
        print(f"  Sample query item: {queries[sample_query_id]}")

        if sample_query_id in qrels:
            print(f"  Sample qrels for query: {qrels[sample_query_id]}")

        return True

    except Exception as e:
        print(f"Error checking BEIR {dataset_name}: {e}")
        return False

def main():
    print("Investigating modern dataset structures...")
    print("=" * 50)

    # Check Cohere datasets
    cohere_datasets = [
        "Cohere/wikipedia-22-12-simple-embeddings",
        # "Cohere/wikipedia-22-12-en-embeddings",  # Too large for quick check
    ]

    for dataset in cohere_datasets:
        print()
        check_huggingface_dataset(dataset)
        print("-" * 30)

    # Check BEIR datasets
    beir_datasets = [
        "fiqa",  # Small dataset
        # "trec-covid",  # Medium dataset
        # "msmarco",  # Large dataset
    ]

    for dataset in beir_datasets:
        print()
        check_beir_dataset(dataset)
        print("-" * 30)

if __name__ == "__main__":
    main()