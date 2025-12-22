#!/usr/bin/env python3
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024-present, Zvi Schneider
#
# Create a dummy dataset for testing the vector search pipeline
#
# This file is part of valkey-search-benchmark and is licensed under the
# BSD 3-Clause License. See the LICENSE file in the root directory.
#

"""
Create a dummy dataset with low dimensions for testing the vector search pipeline end-to-end.
This creates a small, controlled dataset with known ground truth relationships.
"""

import numpy as np
import h5py
import sys
import os

def create_dummy_dataset(output_file, dimensions=8, num_vectors=1000, num_queries=100, k_neighbors=10):
    """
    Create a dummy dataset with controlled properties for testing.

    Args:
        output_file: Path to output HDF5 file
        dimensions: Vector dimensions (default: 8)
        num_vectors: Number of vectors in dataset (default: 1000)
        num_queries: Number of query vectors (default: 100)
        k_neighbors: Number of neighbors per query (default: 10)
    """
    print(f"Creating dummy dataset: {num_vectors} vectors, {dimensions}D")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create base vectors with some structure
    # Use a mix of random and structured patterns to create realistic similarities
    train_vectors = []

    # Create clusters of similar vectors for realistic ground truth
    num_clusters = 10
    vectors_per_cluster = num_vectors // num_clusters

    for cluster_id in range(num_clusters):
        # Create a cluster center
        cluster_center = np.random.randn(dimensions).astype(np.float32)
        cluster_center = cluster_center / np.linalg.norm(cluster_center)  # Normalize

        # Create vectors around this cluster
        for _ in range(vectors_per_cluster):
            # Add some noise to the cluster center
            noise = np.random.normal(0, 0.3, dimensions)
            vector = cluster_center + noise
            vector = vector.astype(np.float32)
            train_vectors.append(vector)

    # Fill remaining vectors if any
    while len(train_vectors) < num_vectors:
        vector = np.random.randn(dimensions).astype(np.float32)
        train_vectors.append(vector)

    train_vectors = np.array(train_vectors[:num_vectors])
    print(f"Created {len(train_vectors)} training vectors")

    # Create query vectors (subset of training vectors for guaranteed matches)
    query_indices = np.random.choice(num_vectors, num_queries, replace=False)
    query_vectors = train_vectors[query_indices].copy()
    print(f"Selected {len(query_vectors)} query vectors")

    # Compute ground truth neighbors for each query
    print("Computing ground truth neighbors...")
    neighbors = np.zeros((num_queries, k_neighbors), dtype=np.int32)

    for i, query_idx in enumerate(query_indices):
        query = query_vectors[i:i+1]

        # Compute cosine similarities
        similarities = np.dot(train_vectors, query.T).flatten()

        # Get top k neighbors (including the query itself)
        top_indices = np.argsort(similarities)[-k_neighbors:][::-1]
        neighbors[i] = top_indices

        if i % 20 == 0:
            print(f"  Computed neighbors for query {i}/{num_queries}")

    # Verify ground truth quality
    print("\nVerifying ground truth quality...")
    perfect_matches = 0
    for i, query_idx in enumerate(query_indices):
        if neighbors[i][0] == query_idx:  # Query should be its own top neighbor
            perfect_matches += 1

    print(f"Perfect self-matches: {perfect_matches}/{num_queries} ({perfect_matches/num_queries*100:.1f}%)")

    # Save to HDF5 file
    print(f"\nSaving to {output_file}...")
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('train', data=train_vectors)
        f.create_dataset('test', data=query_vectors)
        f.create_dataset('neighbors', data=neighbors)

        # Add metadata
        f.attrs['dimensions'] = dimensions
        f.attrs['num_vectors'] = num_vectors
        f.attrs['num_queries'] = num_queries
        f.attrs['k_neighbors'] = k_neighbors
        f.attrs['description'] = 'Dummy dataset for testing vector search pipeline'

    print(f"✓ Successfully created dummy dataset:")
    print(f"  Vectors: {num_vectors} x {dimensions}D")
    print(f"  Queries: {num_queries}")
    print(f"  Neighbors per query: {k_neighbors}")
    print(f"  File size: {os.path.getsize(output_file) / 1024:.1f} KB")

    return True

def verify_dataset(hdf5_file):
    """Verify the created dataset has the expected structure."""
    print(f"\nVerifying dataset: {hdf5_file}")

    with h5py.File(hdf5_file, 'r') as f:
        print("Dataset structure:")
        for key in f.keys():
            dataset = f[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")

        print("\nMetadata:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")

        # Check data ranges
        train = f['train'][:]
        neighbors = f['neighbors'][:]

        print(f"\nData validation:")
        print(f"  Train vectors range: [{train.min():.3f}, {train.max():.3f}]")
        print(f"  Neighbor indices range: [{neighbors.min()}, {neighbors.max()}]")
        print(f"  Max neighbor index should be < {len(train)}: {neighbors.max() < len(train)}")

        # Check some ground truth quality
        print(f"\nGround truth sample (first 5 queries):")
        for i in range(min(5, len(neighbors))):
            print(f"  Query {i}: top 5 neighbors = {neighbors[i][:5]}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_dummy_dataset.py <output_file> [dimensions] [num_vectors] [num_queries]")
        print("Example: python create_dummy_dataset.py dummy-test.hdf5 8 1000 100")
        sys.exit(1)

    output_file = sys.argv[1]
    dimensions = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    num_vectors = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    num_queries = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    k_neighbors = 10  # Fixed for now

    try:
        success = create_dummy_dataset(output_file, dimensions, num_vectors, num_queries, k_neighbors)
        if success:
            verify_dataset(output_file)
            print(f"\n✓ Dummy dataset ready for testing!")
            print(f"Next steps:")
            print(f"  1. Convert to binary: python prep_datasets/prepare_binary.py {output_file} dummy-test.bin")
            print(f"  2. Add to test config: DATASET_CONFIG[\"dummy-test\"]=\"zvec_dummy:,{dimensions},{num_vectors},{k_neighbors}\"")
            print(f"  3. Test: ./test_multi_dataset.sh --dataset dummy-test")
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()