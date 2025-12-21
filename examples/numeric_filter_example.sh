#!/bin/bash
# Example: Vector Search Benchmark with Numeric and Tag Fields
#
# This example demonstrates loading vectors with:
# - Tag field (category) with distribution
# - Numeric field (price) with uniform distribution
# - Then querying with tag filter
#
# The benchmark automatically creates the index with vector, tag, and numeric fields.
#
# Prerequisites:
# - Build: cargo build --release
# - Dataset: mnist.bin (or any compatible binary dataset)
# - ValkeySearch cluster running

set -e

# Configuration
HOST="${HOST:-localhost}"
DATASET="${DATASET:-/data/valkey-bench-glide/valkey-search-bench/datasets/mnist.bin}"
INDEX_NAME="numeric_filter_idx"
PREFIX="vec:"
VECTOR_FIELD="embedding"
TAG_FIELD="category"
PRICE_FIELD="price"

# Binary path (adjust if needed)
BENCH="./target/release/valkey-search-benchmark"

echo "=========================================="
echo "Numeric/Tag Field Example"
echo "=========================================="
echo "Host: $HOST"
echo "Dataset: $DATASET"
echo "Index: $INDEX_NAME"
echo ""

# Step 1: Flush existing data (optional, comment out to keep data)
echo "[Step 1] Flushing existing data..."
$BENCH --cli -h "$HOST" FLUSHALL 2>/dev/null || true

echo ""
echo "=========================================="
echo "Phase 1: Load Vectors with Tag + Numeric Fields"
echo "=========================================="
echo ""
echo "The benchmark will automatically create the index with:"
echo "  - Vector field: $VECTOR_FIELD (HNSW, dim from dataset, L2)"
echo "  - Tag field: $TAG_FIELD"
echo "  - Numeric field: $PRICE_FIELD"
echo ""

# Load 10,000 vectors with:
# - Tag field: category with distribution (electronics:30%, clothing:25%, books:20%, home:15%, toys:10%)
# - Numeric field: price with uniform distribution from $0.99 to $999.99 (2 decimal places)
$BENCH -h "$HOST" --cluster \
    -t vec-load \
    --dataset "$DATASET" \
    --search-prefix "$PREFIX" \
    --search-index "$INDEX_NAME" \
    --search-vector-field "$VECTOR_FIELD" \
    --tag-field "$TAG_FIELD" \
    --search-tags "electronics:30,clothing:25,books:20,home:15,toys:10" \
    --numeric-field-config "$PRICE_FIELD:float:uniform:0.99:999.99:2" \
    -n 10000 \
    -c 50 \
    --threads 4

echo ""
echo "=========================================="
echo "Verify: Check sample records"
echo "=========================================="

# Check first few keys to verify data
echo "Sample keys:"
$BENCH --cli -h "$HOST" KEYS "${PREFIX}*" 2>/dev/null | head -3

# Get fields from a sample record
SAMPLE_KEY=$($BENCH --cli -h "$HOST" KEYS "${PREFIX}*" 2>/dev/null | head -1 | tr -d '"' | sed 's/^[0-9]*) //')
echo ""
echo "Fields in $SAMPLE_KEY:"
$BENCH --cli -h "$HOST" HKEYS "$SAMPLE_KEY" 2>/dev/null || echo "(key not found)"

echo ""
echo "Values (category and price):"
$BENCH --cli -h "$HOST" HMGET "$SAMPLE_KEY" "$TAG_FIELD" "$PRICE_FIELD" 2>/dev/null || echo "(key not found)"

echo ""
echo "=========================================="
echo "Phase 2: Vector Query with Tag Filter"
echo "=========================================="

# Query vectors filtered by tag (only electronics category)
$BENCH -h "$HOST" --cluster \
    -t vec-query \
    --dataset "$DATASET" \
    --search-prefix "$PREFIX" \
    --search-index "$INDEX_NAME" \
    --search-vector-field "$VECTOR_FIELD" \
    --tag-field "$TAG_FIELD" \
    --tag-filter "electronics" \
    -k 10 \
    -n 1000 \
    -c 20 \
    --threads 4

echo ""
echo "=========================================="
echo "Phase 3: Vector Query without Filter (baseline)"
echo "=========================================="

# Query without filter for comparison
$BENCH -h "$HOST" --cluster \
    -t vec-query \
    --dataset "$DATASET" \
    --search-prefix "$PREFIX" \
    --search-index "$INDEX_NAME" \
    --search-vector-field "$VECTOR_FIELD" \
    -k 10 \
    -n 1000 \
    -c 20 \
    --threads 4

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
echo ""
echo "Notes:"
echo "- The 'electronics' filter returns only vectors with category=electronics (~30% of data)"
echo "- Recall may be low since ground truth is computed against full dataset"
echo "- For production, use larger datasets and adjust parameters accordingly"
echo ""
echo "Available numeric field distributions:"
echo "  --numeric-field-config 'name:int:uniform:min:max'"
echo "  --numeric-field-config 'name:float:uniform:min:max:precision'"
echo "  --numeric-field-config 'name:float:normal:mean:stddev:precision'"
echo "  --numeric-field-config 'name:int:zipfian:skew:min:max'"
echo "  --numeric-field-config 'name:unix_timestamp:uniform:min:max'"
echo "  --numeric-field-config 'name:iso_datetime:uniform:min:max'"
echo "  --numeric-field-config 'name:int:sequential:start:step'"
echo "  --numeric-field-config 'name:float:constant:value'"
echo "  --numeric-field-config 'name:int:key_based:min:max'"
