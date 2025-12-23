#!/usr/bin/env python3
"""
Example: Generate a key-value dataset using the CommandRecorder API.

This script demonstrates how developers can use the CommandRecorder to
create custom datasets for benchmarking with valkey-bench-rs.

Usage:
    # Generate 3M keys with 500-byte values
    python create_kv_dataset.py -o datasets/kv_3m -n 3000000 -d 500

    # Generate 1M keys with 100-byte values
    python create_kv_dataset.py -o datasets/kv_1m -n 1000000 -d 100
"""

import argparse
import numpy as np
import time
from pathlib import Path

# Import the CommandRecorder from the same directory
from command_recorder import CommandRecorder, Text, Blob


def create_kv_dataset(output_base: Path, num_keys: int, value_size: int,
                      key_prefix: str = "key:"):
    """
    Create a key-value dataset using CommandRecorder.

    This demonstrates the recommended way for developers to generate
    custom datasets for valkey-bench-rs benchmarking.
    """
    output_base = Path(output_base)

    print(f"Generating key-value dataset using CommandRecorder:")
    print(f"  Keys: {num_keys:,}")
    print(f"  Value size: {value_size} bytes")
    print(f"  Key prefix: {key_prefix}")
    print()

    # Create recorder with dataset name
    rec = CommandRecorder(name=output_base.name)

    # Declare the schema upfront (recommended for large datasets)
    # For SET commands, the value is stored as _arg0 (first positional argument)
    rec.declare_field("_arg0", "blob", max_bytes=value_size)

    # Generate random values
    np.random.seed(42)  # Reproducible results

    start_time = time.time()
    batch_size = 100000

    for batch_start in range(0, num_keys, batch_size):
        batch_end = min(batch_start + batch_size, num_keys)

        # Generate batch of random values
        for i in range(batch_start, batch_end):
            # Generate random bytes for the value
            value = np.random.bytes(value_size)

            # Record a SET command: SET key value
            rec.record("SET", f"{key_prefix}{i:012d}", Blob(value))

        # Progress update
        elapsed = time.time() - start_time
        rate = batch_end / elapsed if elapsed > 0 else 0
        print(f"  Recorded {batch_end:,}/{num_keys:,} commands ({rate:,.0f} keys/sec)")

    print()
    print(rec.summary())
    print()

    # Generate the schema YAML and binary data files
    schema_path, data_path = rec.generate(str(output_base))

    elapsed = time.time() - start_time
    file_size = data_path.stat().st_size

    print()
    print(f"Generation complete:")
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  Size: {file_size:,} bytes ({file_size / (1024**3):.2f} GB)")
    print()
    print(f"Usage:")
    print(f"  # Load data into cluster")
    print(f"  ./target/release/valkey-bench-rs -h HOST --cluster \\")
    print(f"    --schema {schema_path} --data {data_path} \\")
    print(f"    -t set -n {num_keys} -c 100 --threads 16")
    print()
    print(f"  # Run GET benchmark (uses same keyspace)")
    print(f"  ./target/release/valkey-bench-rs -h HOST --cluster \\")
    print(f"    -t get -n {num_keys} -r {num_keys} -c 500 --threads 52")

    return schema_path, data_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate key-value dataset using CommandRecorder API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 3M keys with 500-byte values (standard benchmark)
  python create_kv_dataset.py -o datasets/kv_3m -n 3000000 -d 500

  # Generate smaller test dataset
  python create_kv_dataset.py -o datasets/kv_test -n 10000 -d 100

  # Custom prefix
  python create_kv_dataset.py -o datasets/users -n 100000 -d 256 --prefix "user:"

This script demonstrates how to use the CommandRecorder API to create
custom datasets. Developers can use this pattern to build their own
datasets with specific data patterns, field types, and structures.
"""
    )

    parser.add_argument('-o', '--output', type=Path, required=True,
                        help='Output base path (without extension)')
    parser.add_argument('-n', '--num-keys', type=int, default=3000000,
                        help='Number of keys to generate (default: 3000000)')
    parser.add_argument('-d', '--data-size', type=int, default=500,
                        help='Value size in bytes (default: 500)')
    parser.add_argument('--prefix', type=str, default='key:',
                        help='Key prefix (default: "key:")')

    args = parser.parse_args()

    create_kv_dataset(
        args.output,
        args.num_keys,
        args.data_size,
        args.prefix
    )


if __name__ == '__main__':
    main()
