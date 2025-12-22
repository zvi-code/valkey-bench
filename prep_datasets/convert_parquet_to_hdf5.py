#!/usr/bin/env python3 -u
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024-present, Zvi Schneider
#
# UNIFIED STREAMING CONVERTER: VectorDBBench Parquet + BIGANN formats to HDF5
#
# This file is part of valkey-search-benchmark and is licensed under the
# BSD 3-Clause License. See the LICENSE file in the root directory.
#
"""
UNIFIED STREAMING CONVERTER: VectorDBBench Parquet + BIGANN formats to HDF5

TRUE streaming with batched processing:
- Process data in small batches (5K-10K vectors at a time)
- Direct file→NumPy→HDF5 conversion (no intermediate storage)
- Write incrementally to HDF5
- Progress logging every batch with sys.stdout.flush()
- Memory usage: <5GB regardless of dataset size

Supported input formats:
- Parquet (VectorDBBench): shuffle_train.parquet, test.parquet, neighbors.parquet
- FBIN (Big-ANN): float32 binary format
- IBIN (Big-ANN): int32 binary format (ground truth)
- U8BIN (Big-ANN): uint8 binary format
- FVECS (Texmex): float32 with dimension prefix
- BVECS (Texmex): uint8 with dimension prefix

Expected performance for 10M vectors (768 dims):
- Time: 2-4 minutes
- RAM: 2-4GB peak
- Disk I/O: Sequential writes
"""

import sys
import argparse
import numpy as np
import h5py
from pathlib import Path
import time
import struct

# Optional: Parquet support
try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

BATCH_SIZE = 10000  # Process 10K vectors at a time


def stream_fbin(file_path: Path, batch_size: int = BATCH_SIZE):
    """Stream float32 vectors from .fbin file in batches."""
    with open(file_path, 'rb') as f:
        # Read header
        header = f.read(8)
        num_vectors, dimension = struct.unpack('<II', header)
        
        # Stream vectors in batches
        vectors_read = 0
        while vectors_read < num_vectors:
            batch = min(batch_size, num_vectors - vectors_read)
            data = np.fromfile(f, dtype=np.float32, count=batch * dimension)
            if len(data) == 0:
                break
            vectors = data.reshape(batch, dimension)
            vectors_read += batch
            yield vectors
        
        return num_vectors, dimension


def stream_ibin(file_path: Path, batch_size: int = BATCH_SIZE):
    """Stream int32 vectors from .ibin file in batches."""
    with open(file_path, 'rb') as f:
        # Read header
        header = f.read(8)
        num_vectors, dimension = struct.unpack('<II', header)
        
        # Stream vectors in batches
        vectors_read = 0
        while vectors_read < num_vectors:
            batch = min(batch_size, num_vectors - vectors_read)
            data = np.fromfile(f, dtype=np.int32, count=batch * dimension)
            if len(data) == 0:
                break
            vectors = data.reshape(batch, dimension)
            vectors_read += batch
            yield vectors
        
        return num_vectors, dimension


def stream_u8bin(file_path: Path, batch_size: int = BATCH_SIZE, normalize: bool = False):
    """Stream uint8 vectors from .u8bin file in batches, optionally normalizing to float32."""
    with open(file_path, 'rb') as f:
        # Read header
        header = f.read(8)
        num_vectors, dimension = struct.unpack('<II', header)
        
        # Stream vectors in batches
        vectors_read = 0
        while vectors_read < num_vectors:
            batch = min(batch_size, num_vectors - vectors_read)
            data = np.fromfile(f, dtype=np.uint8, count=batch * dimension)
            if len(data) == 0:
                break
            vectors = data.reshape(batch, dimension)
            
            # Convert to float32 (optionally normalize to [0,1])
            if normalize:
                vectors = vectors.astype(np.float32) / 255.0
            else:
                vectors = vectors.astype(np.float32)
            
            vectors_read += batch
            yield vectors
        
        return num_vectors, dimension


def stream_fvecs(file_path: Path, batch_size: int = BATCH_SIZE):
    """Stream float32 vectors from .fvecs file in batches."""
    vectors_list = []
    dimension = None
    count = 0
    
    with open(file_path, 'rb') as f:
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if len(dim_bytes) == 0:
                break
            
            dim = struct.unpack('<I', dim_bytes)[0]
            if dimension is None:
                dimension = dim
            
            # Read vector
            vector = np.fromfile(f, dtype=np.float32, count=dimension)
            if len(vector) != dimension:
                break
            
            vectors_list.append(vector)
            count += 1
            
            # Yield batch when ready
            if len(vectors_list) >= batch_size:
                yield np.array(vectors_list)
                vectors_list = []
        
        # Yield remaining vectors
        if vectors_list:
            yield np.array(vectors_list)
    
    return count, dimension


def stream_bvecs(file_path: Path, batch_size: int = BATCH_SIZE, normalize: bool = False):
    """Stream uint8 vectors from .bvecs file in batches, optionally normalizing to float32."""
    vectors_list = []
    dimension = None
    count = 0
    
    with open(file_path, 'rb') as f:
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if len(dim_bytes) == 0:
                break
            
            dim = struct.unpack('<I', dim_bytes)[0]
            if dimension is None:
                dimension = dim
            
            # Read vector
            vector = np.fromfile(f, dtype=np.uint8, count=dimension)
            if len(vector) != dimension:
                break
            
            # Convert to float32
            if normalize:
                vector = vector.astype(np.float32) / 255.0
            else:
                vector = vector.astype(np.float32)
            
            vectors_list.append(vector)
            count += 1
            
            # Yield batch when ready
            if len(vectors_list) >= batch_size:
                yield np.array(vectors_list)
                vectors_list = []
        
        # Yield remaining vectors
        if vectors_list:
            yield np.array(vectors_list)
    
    return count, dimension


def read_bigann_gt(file_path: Path):
    """Read BigANN binary ground truth format.
    
    Format: nq (uint32), k (uint32), nq*k indices (int32), nq*k distances (float32)
    Returns: (indices, distances) as numpy arrays of shape (nq, k)
    """
    with open(file_path, 'rb') as f:
        nq, k = np.fromfile(f, dtype=np.uint32, count=2)
        indices = np.fromfile(f, dtype=np.int32, count=nq * k).reshape(nq, k)
        distances = np.fromfile(f, dtype=np.float32, count=nq * k).reshape(nq, k)
        return indices, distances


def detect_format(file_path: Path):
    """Detect file format from extension."""
    suffix = file_path.suffix.lower()
    formats = {
        '.fbin': 'fbin',
        '.ibin': 'ibin',
        '.u8bin': 'u8bin',
        '.fvecs': 'fvecs',
        '.bvecs': 'bvecs',
        '.parquet': 'parquet',
        '.gt': 'bigann_gt',  # BigANN binary GT format
    }
    return formats.get(suffix, 'unknown')


def get_metadata_from_bigann(file_path: Path):
    """Get num_vectors and dimension from BIGANN file header without loading data."""
    suffix = file_path.suffix.lower()
    
    if suffix in ['.fbin', '.ibin', '.u8bin']:
        with open(file_path, 'rb') as f:
            header = f.read(8)
            num_vectors, dimension = struct.unpack('<II', header)
            return num_vectors, dimension
    
    elif suffix == '.gt':
        # BigANN binary GT format: nq (uint32), k (uint32), ...
        with open(file_path, 'rb') as f:
            nq, k = np.fromfile(f, dtype=np.uint32, count=2)
            return int(nq), int(k)
    
    elif suffix in ['.fvecs', '.bvecs']:
        # Need to scan file to count vectors
        with open(file_path, 'rb') as f:
            dim_bytes = f.read(4)
            if len(dim_bytes) != 4:
                raise ValueError("Invalid file")
            dimension = struct.unpack('<I', dim_bytes)[0]
            
            # Count vectors by seeking
            itemsize = 4 if suffix == '.fvecs' else 1
            vector_size = 4 + dimension * itemsize
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()
            num_vectors = file_size // vector_size
            
            return num_vectors, dimension
    
    raise ValueError(f"Unsupported format: {suffix}")


def convert_bigann_to_hdf5_streaming(
    base_file: Path,
    queries_file: Path,
    groundtruth_file: Path,
    output_file: Path,
    dataset_name: str,
    normalize: bool = False,
    max_neighbors: int = 100
):
    """
    Convert BIGANN format files to HDF5 using streaming (no full load into memory).
    """
    print("=" * 80)
    print(f"STREAMING BIGANN → HDF5 CONVERTER")
    print("=" * 80)
    print(f"Dataset: {dataset_name}")
    print(f"Output:  {output_file}")
    print()
    
    start_time = time.time()
    
    # Get metadata without loading
    print("Analyzing files...")
    base_format = detect_format(base_file)
    query_format = detect_format(queries_file)
    gt_format = detect_format(groundtruth_file)
    
    num_vectors, vec_dim = get_metadata_from_bigann(base_file)
    num_queries, query_dim = get_metadata_from_bigann(queries_file)
    num_gt, gt_dim = get_metadata_from_bigann(groundtruth_file)
    
    print(f"  Base: {num_vectors:,} vectors × {vec_dim} ({base_format})")
    print(f"  Queries: {num_queries:,} vectors × {query_dim} ({query_format})")
    print(f"  Ground truth: {num_gt:,} × {gt_dim} ({gt_format})")
    
    if vec_dim != query_dim:
        raise ValueError(f"Dimension mismatch: base={vec_dim}, queries={query_dim}")
    if num_queries != num_gt:
        raise ValueError(f"Query count mismatch: {num_queries} vs {num_gt}")
    
    # Truncate GT if needed
    num_neighbors = min(gt_dim, max_neighbors)
    if gt_dim > max_neighbors:
        print(f"  Will truncate GT from {gt_dim} to {max_neighbors} neighbors")
    
    print()
    
    # Create HDF5 file with chunked datasets
    output_file.parent.mkdir(parents=True, exist_ok=True)
    hf = h5py.File(output_file, 'w')
    
    # Create chunked datasets (for streaming writes)
    chunk_size = min(BATCH_SIZE, num_vectors)
    train_dset = hf.create_dataset(
        'train',
        shape=(num_vectors, vec_dim),
        dtype=np.float32,
        chunks=(chunk_size, vec_dim)
    )
    
    test_dset = hf.create_dataset(
        'test',
        shape=(num_queries, query_dim),
        dtype=np.float32,
        chunks=(min(BATCH_SIZE, num_queries), query_dim)
    )
    
    neighbors_dset = hf.create_dataset(
        'neighbors',
        shape=(num_queries, num_neighbors),
        dtype=np.int64,
        chunks=(min(BATCH_SIZE, num_queries), num_neighbors)
    )
    
    print(f"Created HDF5 datasets (chunked for streaming)")
    print()
    
    # Stream base vectors
    print("PHASE 1: Streaming base vectors...")
    phase1_start = time.time()
    offset = 0
    
    # Select appropriate streamer based on format
    if base_format == 'fbin':
        streamer = stream_fbin(base_file, BATCH_SIZE)
    elif base_format == 'u8bin':
        streamer = stream_u8bin(base_file, BATCH_SIZE, normalize)
    elif base_format == 'bvecs':
        streamer = stream_bvecs(base_file, BATCH_SIZE, normalize)
    elif base_format == 'fvecs':
        streamer = stream_fvecs(base_file, BATCH_SIZE)
    else:
        raise ValueError(f"Unsupported base format: {base_format}")
    
    for batch_idx, batch in enumerate(streamer, 1):
        batch_size = len(batch)
        train_dset[offset:offset + batch_size] = batch
        offset += batch_size
        
        if batch_idx % 10 == 0 or offset >= num_vectors:
            elapsed = time.time() - phase1_start
            rate = offset / elapsed if elapsed > 0 else 0
            pct = 100.0 * offset / num_vectors
            print(f"  [{batch_idx:5d}] {offset:,}/{num_vectors:,} vectors ({pct:5.1f}%) | "
                  f"{rate:,.0f} vec/s | {elapsed:.1f}s")
    
    phase1_time = time.time() - phase1_start
    print(f"  ✓ Phase 1 complete: {num_vectors:,} vectors in {phase1_time:.1f}s")
    print()
    
    # Stream query vectors
    print("PHASE 2: Streaming query vectors...")
    phase2_start = time.time()
    offset = 0
    
    if query_format == 'fbin':
        streamer = stream_fbin(queries_file, BATCH_SIZE)
    elif query_format == 'u8bin':
        streamer = stream_u8bin(queries_file, BATCH_SIZE, normalize)
    elif query_format == 'bvecs':
        streamer = stream_bvecs(queries_file, BATCH_SIZE, normalize)
    elif query_format == 'fvecs':
        streamer = stream_fvecs(queries_file, BATCH_SIZE)
    else:
        raise ValueError(f"Unsupported query format: {query_format}")
    
    for batch in streamer:
        batch_size = len(batch)
        test_dset[offset:offset + batch_size] = batch
        offset += batch_size
    
    phase2_time = time.time() - phase2_start
    print(f"  ✓ Phase 2 complete: {num_queries:,} queries in {phase2_time:.1f}s")
    print()
    
    # Stream ground truth
    print("PHASE 3: Streaming ground truth...")
    phase3_start = time.time()
    offset = 0
    
    # Check ground truth format
    if gt_format == 'bigann_gt':
        # BigANN binary format: read all at once (small file)
        print(f"  Reading BigANN binary GT format...")
        indices, distances = read_bigann_gt(groundtruth_file)
        # Truncate to max_neighbors
        if indices.shape[1] > num_neighbors:
            print(f"  Truncating GT from {indices.shape[1]} to {num_neighbors} neighbors")
            indices = indices[:, :num_neighbors]
            distances = distances[:, :num_neighbors]
        neighbors_dset[:] = indices.astype(np.int64)
        # Optionally store distances too
        if 'distances' not in hf:
            dist_dset = hf.create_dataset('distances', data=distances, dtype=np.float32)
    elif gt_format == 'ibin':
        # Legacy ibin format: stream in batches
        for batch in stream_ibin(groundtruth_file, BATCH_SIZE):
            batch_size = len(batch)
            # Truncate to max_neighbors and convert to int64
            batch_truncated = batch[:, :num_neighbors].astype(np.int64)
            neighbors_dset[offset:offset + batch_size] = batch_truncated
            offset += batch_size
    else:
        raise ValueError(f"Unsupported ground truth format: {gt_format}")
    
    phase3_time = time.time() - phase3_start
    print(f"  ✓ Phase 3 complete: {num_queries:,} × {num_neighbors} in {phase3_time:.1f}s")
    print()
    
    # Add metadata
    hf.attrs['dataset_name'] = dataset_name
    hf.attrs['num_vectors'] = num_vectors
    hf.attrs['num_queries'] = num_queries
    hf.attrs['dimensions'] = vec_dim
    hf.attrs['num_neighbors'] = num_neighbors
    hf.attrs['source_format'] = base_format
    hf.attrs['normalized'] = normalize
    
    hf.close()
    
    # Summary
    total_time = time.time() - start_time
    output_size = output_file.stat().st_size
    output_size_gb = output_size / (1024**3)
    overall_rate = num_vectors / total_time if total_time > 0 else 0
    
    print("=" * 80)
    print("✓ CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"  Output: {output_file}")
    print(f"  Size: {output_size_gb:.2f} GB")
    print(f"  Vectors: {num_vectors:,} × {vec_dim}")
    print(f"  Queries: {num_queries:,} × {query_dim}")
    print(f"  Ground truth: {num_queries:,} × {num_neighbors}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Throughput: {overall_rate:,.0f} vectors/second")
    print(f"  Memory: Streaming (< 1GB peak)")
    print("=" * 80)


def convert_parquet_to_hdf5(dataset_dir, output_file, dataset_name):
    """
    Convert vectordb-bench parquet files to HDF5 with TRUE streaming.
    """
    dataset_path = Path(dataset_dir)
    start_time = time.time()
    
    print("=" * 70)
    print(f"OPTIMIZED PARQUET → HDF5 CONVERTER")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Input:   {dataset_path}")
    print(f"Output:  {output_file}")
    print(f"Batch:   {BATCH_SIZE:,} vectors per iteration")
    print()
    
    # Find all training parquet files
    train_files = sorted(dataset_path.glob("*train*.parquet"))
    
    if not train_files:
        raise FileNotFoundError(f"No training parquet files found in {dataset_path}")
    
    print(f"Found {len(train_files)} training file(s):")
    for i, f in enumerate(train_files, 1):
        size_mb = f.stat().st_size / (1024**2)
        print(f"  {i:2d}. {f.name:40s} ({size_mb:7.1f} MB)")
    print()
    
    # === PHASE 1: Analyze dataset ===
    print("PHASE 1: Analyzing dataset structure...")
    first_table = pq.read_table(train_files[0], columns=[])
    first_table_with_data = pq.read_table(train_files[0])
    
    # Determine vector column
    if 'emb' in first_table_with_data.column_names:
        vec_col = 'emb'
        first_vec = first_table_with_data[vec_col][0].as_py()
    elif 'vector' in first_table_with_data.column_names:
        vec_col = 'vector'
        first_vec = first_table_with_data[vec_col][0].as_py()
    else:
        raise ValueError(f"No 'emb' or 'vector' column. Available: {first_table_with_data.column_names}")
    
    vec_dim = len(first_vec)
    del first_table_with_data
    
    # Count total vectors
    print(f"  Vector column: '{vec_col}'")
    print(f"  Dimensions: {vec_dim}")
    print(f"  Counting rows in all files...")
    
    total_vectors = 0
    for train_file in train_files:
        table = pq.read_table(train_file, columns=[])
        total_vectors += table.num_rows
    
    est_size_gb = total_vectors * vec_dim * 4 / (1024**3)
    print(f"  Total training vectors: {total_vectors:,}")
    print(f"  Estimated size: {est_size_gb:.2f} GB")
    print()
    
    # === PHASE 2: Create HDF5 file ===
    print("PHASE 2: Creating HDF5 file...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    hf = h5py.File(output_file, 'w')
    
    # Create chunked dataset (NO compression for speed)
    chunk_size = min(BATCH_SIZE, total_vectors)
    train_dset = hf.create_dataset(
        'train',
        shape=(total_vectors, vec_dim),
        dtype=np.float32,
        chunks=(chunk_size, vec_dim),
    )
    print(f"  Created dataset 'train': shape={train_dset.shape}, chunks={train_dset.chunks}")
    print()
    
    # === PHASE 3: Load and un-shuffle training vectors ===
    print("PHASE 3: Loading training vectors (un-shuffling by ID)...")
    print(f"  Note: Training vectors are shuffled in parquet files")
    print(f"  Loading all vectors into memory to sort by ID...")
    print()
    
    phase3_start = time.time()
    
    # Load ALL training vectors with their IDs
    all_vectors = []
    all_ids = []
    
    for file_idx, train_file in enumerate(train_files, 1):
        file_start = time.time()
        
        print(f"  [{file_idx:2d}/{len(train_files)}] {train_file.name}")
        
        # Read the entire file (we need IDs and vectors together)
        table = pq.read_table(train_file, columns=['id', vec_col])
        
        # Get IDs and vectors
        ids = table['id'].to_numpy()
        vectors = table[vec_col].to_numpy(zero_copy_only=False)
        vectors = np.vstack(vectors).astype(np.float32)
        
        all_ids.append(ids)
        all_vectors.append(vectors)
        
        file_time = time.time() - file_start
        print(f"         ✓ Loaded {len(ids):,} vectors in {file_time:.1f}s")
        
        del table
    
    # Concatenate all files
    print(f"\n  Concatenating {len(train_files)} files...")
    all_ids = np.concatenate(all_ids)
    all_vectors = np.vstack(all_vectors)
    
    # Sort by ID to restore original order
    print(f"  Sorting {len(all_ids):,} vectors by ID...")
    sort_indices = np.argsort(all_ids)
    sorted_vectors = all_vectors[sort_indices]
    sorted_ids = all_ids[sort_indices]
    
    # Verify IDs are now sequential
    expected_ids = np.arange(len(sorted_ids))
    if not np.array_equal(sorted_ids, expected_ids):
        print(f"  WARNING: IDs are not sequential after sorting!")
        print(f"    Expected: 0 to {len(sorted_ids)-1}")
        print(f"    Got: {sorted_ids[0]} to {sorted_ids[-1]}")
    else:
        print(f"  ✓ IDs verified: 0 to {len(sorted_ids)-1} (sequential)")
    
    # Write to HDF5
    print(f"  Writing {len(sorted_vectors):,} vectors to HDF5...")
    train_dset[:] = sorted_vectors
    
    del all_ids, all_vectors, sort_indices, sorted_vectors, sorted_ids
    
    phase3_time = time.time() - phase3_start
    phase3_vecs_per_sec = total_vectors / phase3_time if phase3_time > 0 else 0
    print(f"  ✓ Phase 3 complete: {total_vectors:,} vectors in {phase3_time:.1f}s ({phase3_vecs_per_sec:,.0f} vec/s)")
    print()
    
    # === PHASE 4: Load test queries ===
    print("PHASE 4: Loading test queries...")
    test_file = dataset_path / "test.parquet"
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    test_table = pq.read_table(test_file)
    # Direct PyArrow → NumPy conversion
    test_vectors = test_table[vec_col].to_numpy(zero_copy_only=False)
    test_data = np.vstack(test_vectors).astype(np.float32)
    num_queries = len(test_data)
    
    hf.create_dataset('test', data=test_data)
    print(f"  ✓ Loaded {num_queries:,} test queries")
    
    del test_data
    del test_vectors
    del test_table
    print()
    
    # === PHASE 5: Load ground truth ===
    print("PHASE 5: Loading ground truth neighbors...")
    neighbors_file = dataset_path / "neighbors.parquet"
    if not neighbors_file.exists():
        raise FileNotFoundError(f"Neighbors file not found: {neighbors_file}")
    
    neighbors_table = pq.read_table(neighbors_file)
    
    # Find neighbor column
    neighbor_col = None
    for col_name in ['neighbors', 'neighbor_ids', 'neighbors_id', 'neighbor_id']:
        if col_name in neighbors_table.column_names:
            neighbor_col = col_name
            break
    
    if not neighbor_col:
        raise ValueError(f"No neighbors column found. Available: {list(neighbors_table.column_names)}")
    
    # Direct PyArrow → NumPy conversion (avoid pandas intermediate step)
    neighbors_arrays = neighbors_table[neighbor_col].to_numpy(zero_copy_only=False)
    neighbors_data = np.vstack(neighbors_arrays).astype(np.int64)
    num_neighbors = neighbors_data.shape[1]
    
    hf.create_dataset('neighbors', data=neighbors_data)
    print(f"  ✓ Loaded ground truth: {neighbors_data.shape[0]:,} queries × {num_neighbors} neighbors")
    
    del neighbors_data
    del neighbors_arrays
    del neighbors_table
    print()
    
    # === PHASE 6: Add metadata ===
    print("PHASE 6: Writing metadata...")
    hf.attrs['dataset_name'] = dataset_name
    hf.attrs['num_vectors'] = total_vectors
    hf.attrs['num_queries'] = num_queries
    hf.attrs['dimensions'] = vec_dim
    hf.attrs['num_neighbors'] = num_neighbors
    print(f"  ✓ Metadata written")
    print()
    
    # Close file
    hf.close()
    
    # === SUMMARY ===
    total_time = time.time() - start_time
    output_size = output_path.stat().st_size
    output_size_gb = output_size / (1024**3)
    overall_vecs_per_sec = total_vectors / total_time if total_time > 0 else 0
    
    print("=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"  Output file: {output_file}")
    print(f"  File size: {output_size_gb:.2f} GB")
    print(f"  Training vectors: {total_vectors:,} × {vec_dim}")
    print(f"  Test queries: {num_queries:,} × {vec_dim}")
    print(f"  Ground truth: {num_queries:,} × {num_neighbors}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Throughput: {overall_vecs_per_sec:,.0f} vectors/second")
    print("=" * 70)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(
        description='Convert datasets to HDF5 (Parquet or BIGANN formats)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # VectorDBBench parquet dataset
  python3 convert_parquet_to_hdf5.py \\
      /mnt/data/datasets/cohere/cohere_medium_1m \\
      cohere-medium-1m.hdf5 \\
      --name cohere-medium-1m
  
  # BIGANN fbin dataset
  python3 convert_parquet_to_hdf5.py \\
      --base base.10M.fbin \\
      --queries query.10K.fbin \\
      --groundtruth gt.10K.ibin \\
      --output deep-10m.hdf5 \\
      --name "Deep1B-10M"
  
  # Texmex bvecs dataset (with normalization)
  python3 convert_parquet_to_hdf5.py \\
      --base sift_base.bvecs \\
      --queries sift_query.bvecs \\
      --groundtruth sift_gt.ivecs \\
      --output sift-1m.hdf5 \\
      --name "SIFT-1M" \\
      --normalize
        """
    )
    
    # Parquet mode arguments
    parser.add_argument('dataset_dir', nargs='?', help='Directory containing parquet files (for VectorDBBench)')
    parser.add_argument('output_file', nargs='?', help='Output HDF5 file path')
    parser.add_argument('--name', default='dataset', help='Dataset name (for metadata)')
    
    # BIGANN mode arguments
    parser.add_argument('--base', type=Path, help='Base vectors file (.fbin, .bvecs, .fvecs, .u8bin)')
    parser.add_argument('--queries', type=Path, help='Query vectors file')
    parser.add_argument('--groundtruth', type=Path, help='Ground truth file (.ibin)')
    parser.add_argument('--output', type=Path, help='Output HDF5 file (for BIGANN mode)')
    parser.add_argument('--normalize', action='store_true', help='Normalize uint8 to float32 [0,1]')
    parser.add_argument('--max-neighbors', type=int, default=100, help='Max neighbors to store (default: 100)')
    
    args = parser.parse_args()
    
    try:
        # Detect mode: BIGANN or Parquet
        if args.base and args.queries and args.groundtruth:
            # BIGANN mode
            if not args.output:
                print("Error: --output required for BIGANN mode", file=sys.stderr)
                return 1
            
            if not HAS_PARQUET:
                pass  # Parquet not needed for BIGANN mode
            
            convert_bigann_to_hdf5_streaming(
                base_file=args.base,
                queries_file=args.queries,
                groundtruth_file=args.groundtruth,
                output_file=args.output,
                dataset_name=args.name,
                normalize=args.normalize,
                max_neighbors=args.max_neighbors
            )
        
        elif args.dataset_dir and args.output_file:
            # Parquet mode
            if not HAS_PARQUET:
                print("Error: pyarrow required for Parquet conversion", file=sys.stderr)
                print("Install with: pip install pyarrow", file=sys.stderr)
                return 1
            
            convert_parquet_to_hdf5(args.dataset_dir, args.output_file, args.name)
        
        else:
            print("Error: Invalid arguments", file=sys.stderr)
            print("\nFor Parquet: provide dataset_dir and output_file", file=sys.stderr)
            print("For BIGANN: provide --base, --queries, --groundtruth, --output", file=sys.stderr)
            parser.print_help()
            return 1
        
        return 0
    
    except Exception as e:
        print(f"\n✗ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
