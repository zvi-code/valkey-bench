# Dataset Conversion Scripts

Generic tools for converting vector datasets to Valkey binary format.

## Overview

This directory contains format-agnostic conversion utilities used by the unified dataset manager.

**For end users**: Use `./prep_datasets/dataset.sh` (from project root) for all dataset operations (download, convert, verify).

## Tools

### download_dataset.py

Generic downloader for VectorDBBench datasets.

```bash
python download_dataset.py COHERE 1000000
python download_dataset.py OPENAI 5000000
```

Requires `vectordb_bench` library. Typically called by `dataset.sh`, not used directly.

### convert_parquet_to_hdf5.py

Generic Parquet → HDF5 converter with memory-efficient chunked processing.

```bash
python convert_parquet_to_hdf5.py \
    /path/to/parquet_dir \
    output.hdf5 \
    --name dataset-name
```

**Features:**
- Low memory footprint for large datasets
- Generates HDF5 with `/train`, `/test`, `/neighbors`, `/distances` structure
- Compatible with `prep_datasets/prepare_binary.py`

Typically called by `dataset.sh`, not used directly.

## Conversion Pipeline

All datasets follow this pipeline:

```
Source Format → HDF5 (intermediate) → Binary (.bin)
     ↓              ↓                     ↓
  Download      Convert HDF5         prepare_binary.py
                                     (in prep_datasets/)
```

### prepare_binary.py

Converts HDF5 format to Valkey binary format.

```bash
python prepare_binary.py input.hdf5 output.bin --metric L2 --max-neighbors 100
```

## Binary Format

Final output is Valkey binary format (`.bin`):
- Magic: `VDSET001`
- Contains: vectors, queries, ground truth, optional metadata
- Used by: `./target/release/valkey-bench-rs --dataset <name>.bin`

## For End Users

**Recommended workflow**:

```bash
# From project root
./prep_datasets/dataset.sh list              # Show all available datasets
./prep_datasets/dataset.sh get mnist         # Download + convert
./prep_datasets/dataset.sh verify file.bin   # Verify integrity
```

See [../DATASETS.md](../DATASETS.md) for complete documentation.
