# Example Datasets

This folder contains sample datasets demonstrating the schema-driven dataset format and how to create custom datasets using the CommandRecorder API.

## Included Examples

### Vector Search Datasets

| File | Description | Records | Fields |
|------|-------------|---------|--------|
| `test_small.yaml/.bin` | Minimal vector dataset for testing | 100 vectors | 32-dim embedding |
| `test_hash.yaml/.bin` | Hash dataset with vector + metadata | 100 records | 128-dim embedding, category (tag), price (numeric) |
| `product_catalog.yaml/.bin` | E-commerce product embeddings | 100 products | 128-dim embedding, category (tag), name (text), price (numeric) |

### Key-Value Datasets

| File | Description | Records | Fields |
|------|-------------|---------|--------|
| `kv_test.yaml/.bin` | Small key-value dataset for testing | 10,000 keys | 100-byte values |

### Python Examples

| File | Description |
|------|-------------|
| `create_kv_dataset.py` | Example script using CommandRecorder API to generate custom datasets |

## Usage

### Vector Search with Schema-Driven Format

```bash
# Load vectors using schema + data files
./target/release/valkey-bench-rs -h HOST --cluster \
  --schema examples/test_small.yaml \
  --data examples/test_small.bin \
  -t vec-load -n 100 -c 10

# Query vectors (uses query section from schema)
./target/release/valkey-bench-rs -h HOST --cluster \
  --schema examples/test_small.yaml \
  --data examples/test_small.bin \
  -t vec-query -n 10 -c 10 --search-index test_idx
```

### E-commerce Example

```bash
# Load product catalog with embeddings, categories, and prices
./target/release/valkey-bench-rs -h HOST --cluster \
  --schema examples/product_catalog.yaml \
  --data examples/product_catalog.bin \
  -t vec-load -n 100 -c 10 --search-index products

# Create the index (required for filtered search)
./target/release/valkey-bench-rs --cli -h HOST \
  "FT.CREATE products ON HASH PREFIX 1 product: SCHEMA \
   embedding VECTOR HNSW 6 TYPE FLOAT32 DIM 128 DISTANCE_METRIC COSINE \
   category TAG \
   price NUMERIC"

# Query with tag filter
./target/release/valkey-bench-rs -h HOST --cluster \
  --schema examples/product_catalog.yaml \
  --data examples/product_catalog.bin \
  -t vec-query --search-index products \
  --tag-field category --tag-filter "electronics" \
  -n 10 -c 10
```

### Key-Value Benchmark

```bash
# Load key-value pairs using recorded dataset
./target/release/valkey-bench-rs -h HOST --cluster \
  --schema examples/kv_test.yaml \
  --data examples/kv_test.bin \
  -t set -n 10000 -c 50

# Run GET benchmark on same keyspace
./target/release/valkey-bench-rs -h HOST --cluster \
  -t get -n 10000 -r 10000 -c 100
```

## Creating Custom Datasets

Use the CommandRecorder API to create your own datasets:

```bash
# Generate a 3M key dataset with 500-byte values
cd prep_datasets
python create_kv_dataset.py -o ../datasets/my_kv -n 3000000 -d 500
```

See `create_kv_dataset.py` for a complete example of using the CommandRecorder API.

### CommandRecorder API Example

```python
from command_recorder import CommandRecorder, Vector, Tag, Blob, Numeric

# Create recorder
rec = CommandRecorder(name="my_dataset")

# Declare schema (optional but recommended for large datasets)
rec.declare_field("embedding", "vector", dim=128, dtype="float32")
rec.declare_field("category", "tag", max_bytes=32)
rec.declare_field("price", "numeric", dtype="float64")

# Record commands
for i in range(1000):
    vec = np.random.rand(128).astype(np.float32)
    rec.record("HSET", f"product:{i:06d}",
               "embedding", Vector(vec),
               "category", Tag("electronics"),
               "price", Numeric(99.99))

# Generate schema YAML + binary data
rec.generate("output_path")  # Creates output_path.yaml and output_path.bin
```

## Schema Format Reference

See [DATASETS.md](../DATASETS.md) for complete schema format documentation.

### Basic Structure

```yaml
version: 1
metadata:
  name: dataset_name
  description: Human-readable description

# For recorded command replay (SET, HSET, etc.)
replay:
  command: HSET  # or SET

record:
  fields:
  - name: field_name
    type: vector|tag|text|numeric|blob
    # Type-specific options...

sections:
  records:
    count: 1000
  keys:
    present: true|false
    pattern: "prefix:{id}"  # Key pattern template
  queries:
    present: true|false
    count: 100
  ground_truth:
    present: true|false
    neighbors_per_query: 10

field_metadata:
  embedding:
    distance_metric: l2|cosine|ip
  category:
    index_type: tag
```

## See Also

- [DATASETS.md](../DATASETS.md) - Complete dataset format documentation
- [EXAMPLES.md](../EXAMPLES.md) - More benchmark examples
- [prep_datasets/command_recorder.py](../prep_datasets/command_recorder.py) - Full CommandRecorder API
