#!/usr/bin/env python3
import numpy as np
import h5py
import struct
from pathlib import Path
import sys
import argparse

DATASET_MAGIC = 0xDECDB001
DATASET_VERSION = 1
DISTANCE_L2 = 0
DISTANCE_COSINE = 1
DISTANCE_IP = 2

def align_to(size, alignment=64):
    return ((size + alignment - 1) // alignment) * alignment

def prepare_dataset_binary(h5_path, output_path, dataset_name,
                          distance_metric=DISTANCE_L2, max_ground_truth=100):
    print(f"Loading dataset from {h5_path}...")

    with h5py.File(h5_path, 'r') as f:
        # Load vectors (train set)
        if 'train' in f:
            vectors = np.array(f['train'], dtype=np.float32)
        elif 'database' in f:
            vectors = np.array(f['database'], dtype=np.float32)
        else:
            raise ValueError("No 'train' or 'database' key found in HDF5 file")

        # Load queries (test set)
        if 'test' in f:
            queries = np.array(f['test'], dtype=np.float32)
        elif 'queries' in f:
            queries = np.array(f['queries'], dtype=np.float32)
        else:
            raise ValueError("No 'test' or 'queries' key found in HDF5 file")

        # Load or compute ground truth
        if 'neighbors' in f:
            ground_truth = np.array(f['neighbors'], dtype=np.int64)
            if ground_truth.shape[1] > max_ground_truth:
                print(f"Truncating ground truth from {ground_truth.shape[1]} to {max_ground_truth} neighbors")
                ground_truth = ground_truth[:, :max_ground_truth]
        elif 'ground_truth' in f:
            ground_truth = np.array(f['ground_truth'], dtype=np.int64)
            if ground_truth.shape[1] > max_ground_truth:
                print(f"Truncating ground truth from {ground_truth.shape[1]} to {max_ground_truth} neighbors")
                ground_truth = ground_truth[:, :max_ground_truth]
        else:
            print(f"Computing ground truth (k={max_ground_truth}, this may take a while)...")
            ground_truth = compute_ground_truth(vectors, queries, k=max_ground_truth)

    num_vectors, dim = vectors.shape
    num_queries = len(queries)
    num_neighbors = ground_truth.shape[1]

    print(f"Dataset: {num_vectors} vectors, {num_queries} queries, dim={dim}, k={num_neighbors}")

    # Calculate offsets (64-byte aligned)
    header_size = 4096
    vectors_offset = header_size
    vectors_size = align_to(num_vectors * dim * 4)

    queries_offset = vectors_offset + vectors_size
    queries_size = align_to(num_queries * dim * 4)

    ground_truth_offset = queries_offset + queries_size
    ground_truth_size = num_queries * num_neighbors * 8

    total_size = ground_truth_offset + ground_truth_size
    print(f"Output size: {total_size / (1024**3):.2f} GB")

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write binary file
    with open(output_path, 'wb') as f:
        # Prepare dataset name (truncate or pad to 256 bytes)
        name_bytes = dataset_name.encode('utf-8')[:256]
        name_bytes = name_bytes.ljust(256, b'\x00')

        # Calculate padding needed to reach 4096 bytes
        base_header_size = struct.calcsize('<II256sBBxxIQQIxxxxQQQ')
        padding_size = 4096 - base_header_size

        # Write header (4KB)
        header = struct.pack(
            f'<II256sBBxxIQQIxxxxQQQ{padding_size}x',
            DATASET_MAGIC,
            DATASET_VERSION,
            name_bytes,
            distance_metric,
            0,  # dtype: FLOAT32
            dim,
            num_vectors,
            num_queries,
            num_neighbors,
            vectors_offset,
            queries_offset,
            ground_truth_offset
        )
        assert len(header) == 4096, f"Header size mismatch: {len(header)} != 4096"
        f.write(header)

        # Write vectors
        f.seek(vectors_offset)
        vectors.tofile(f)
        padding_size = vectors_size - (num_vectors * dim * 4)
        if padding_size > 0:
            f.write(b'\x00' * padding_size)

        # Write queries
        f.seek(queries_offset)
        queries.tofile(f)
        padding_size = queries_size - (num_queries * dim * 4)
        if padding_size > 0:
            f.write(b'\x00' * padding_size)

        # Write ground truth
        f.seek(ground_truth_offset)
        ground_truth.tofile(f)

    print(f"✓ Dataset written to {output_path}")
    verify_dataset(output_path)

def compute_ground_truth(vectors, queries, k=100):
    """Brute force k-NN ground truth computation"""
    try:
        from scipy.spatial.distance import cdist
    except ImportError:
        print("scipy not found, trying sklearn...")
        from sklearn.metrics.pairwise import euclidean_distances
        cdist = lambda x, y, metric: euclidean_distances(x, y)

    print(f"Computing distances for {len(queries)} queries...")
    distances = cdist(queries, vectors, metric='euclidean')

    print(f"Finding top-{k} neighbors...")
    neighbors = np.argsort(distances, axis=1)[:, :k]

    return neighbors.astype(np.int64)

def verify_dataset(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        assert magic == DATASET_MAGIC, f"Invalid magic: 0x{magic:x}"

        f.seek(0)
        header_data = f.read(4096)
        base_header_size = struct.calcsize('<II256sBBxxIQQIxxxxQQQ')
        padding_size = 4096 - base_header_size
        header_unpacked = struct.unpack(f'<II256sBBxxIQQIxxxxQQQ{padding_size}x', header_data)

        print("✓ Verification passed")
        print(f"  Magic: 0x{header_unpacked[0]:x}")
        print(f"  Version: {header_unpacked[1]}")
        print(f"  Name: {header_unpacked[2].rstrip(b'\\x00').decode('utf-8')}")
        print(f"  Dimensions: {header_unpacked[5]}")
        print(f"  Vectors: {header_unpacked[6]}")
        print(f"  Queries: {header_unpacked[7]}")
        print(f"  Neighbors: {header_unpacked[8]}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert HDF5 dataset to binary format for valkey-benchmark'
    )
    parser.add_argument('input', help='Input HDF5 file')
    parser.add_argument('output', help='Output binary file')
    parser.add_argument('--name', help='Dataset name (default: input filename stem)')
    parser.add_argument('--metric', choices=['L2', 'COSINE', 'IP'],
                       default='L2', help='Distance metric (default: L2)')
    parser.add_argument('--max-neighbors', type=int, default=100,
                       help='Maximum number of ground truth neighbors (default: 100)')

    args = parser.parse_args()

    metric_map = {
        'L2': DISTANCE_L2,
        'COSINE': DISTANCE_COSINE,
        'IP': DISTANCE_IP
    }

    name = args.name or Path(args.input).stem

    prepare_dataset_binary(
        args.input, args.output,
        name,
        metric_map[args.metric],
        args.max_neighbors
    )

if __name__ == '__main__':
    main()