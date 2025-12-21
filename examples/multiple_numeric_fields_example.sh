#!/bin/bash
# Example: Vector Search with Multiple Numeric Fields
#
# This example demonstrates loading vectors with multiple numeric fields
# using different value types and distributions:
# - price: Float with uniform distribution
# - quantity: Integer with zipfian distribution (skewed)
# - rating: Float with normal distribution
# - created_at: Unix timestamp with uniform distribution
#
# This simulates an e-commerce product catalog with vector embeddings.
# The benchmark automatically creates the index with all required fields.

set -e

# Configuration
HOST="${HOST:-localhost}"
DATASET="${DATASET:-/data/valkey-bench-glide/valkey-search-bench/datasets/mnist.bin}"
INDEX_NAME="multi_numeric_idx"
PREFIX="product:"

BENCH="./target/release/valkey-search-benchmark"

echo "=========================================="
echo "Multiple Numeric Fields Example"
echo "=========================================="
echo "Host: $HOST"
echo "Index: $INDEX_NAME"
echo ""

# Step 1: Clean up existing data
echo "[Step 1] Flushing existing data..."
$BENCH --cli -h "$HOST" FLUSHALL 2>/dev/null || true

echo ""
echo "=========================================="
echo "Loading with Multiple Numeric Distributions"
echo "=========================================="
echo ""
echo "The benchmark will automatically create the index with:"
echo "  - Vector field: embedding (HNSW, dim from dataset, L2)"
echo "  - Tag field: category"
echo "  - Numeric fields: price, quantity, rating, created_at"
echo ""
echo "Fields being generated:"
echo "  - category: Tag with distribution (electronics:40%, accessories:30%, cables:30%)"
echo "  - price: Float, uniform [9.99, 499.99], 2 decimals"
echo "  - quantity: Integer, zipfian (skew=1.5) [1, 1000]"
echo "  - rating: Float, normal (mean=4.0, stddev=0.5), 1 decimal"
echo "  - created_at: Unix timestamp, uniform [2023-01-01, 2024-12-31]"
echo ""

# Load vectors with multiple numeric fields
$BENCH -h "$HOST" --cluster \
    -t vec-load \
    --dataset "$DATASET" \
    --search-prefix "$PREFIX" \
    --search-index "$INDEX_NAME" \
    --search-vector-field embedding \
    --tag-field category \
    --search-tags "electronics:40,accessories:30,cables:30" \
    --numeric-field-config "price:float:uniform:9.99:499.99:2" \
    --numeric-field-config "quantity:int:zipfian:1.5:1:1000" \
    --numeric-field-config "rating:float:normal:4.0:0.5:1" \
    --numeric-field-config "created_at:unix_timestamp:uniform:1672531200:1735689600" \
    -n 5000 \
    -c 50 \
    --threads 4

echo ""
echo "=========================================="
echo "Verify: Sample Product Data"
echo "=========================================="

# Get a sample record
SAMPLE_KEY=$($BENCH --cli -h "$HOST" KEYS "${PREFIX}*" 2>/dev/null | head -1 | tr -d '"' | sed 's/^[0-9]*) //')
echo "Sample record: $SAMPLE_KEY"
echo ""

# Show all field values
for field in category price quantity rating created_at; do
    value=$($BENCH --cli -h "$HOST" HGET "$SAMPLE_KEY" "$field" 2>/dev/null | head -1 || echo "N/A")
    printf "  %-12s: %s\n" "$field" "$value"
done

echo ""
echo "=========================================="
echo "Run Vector Queries"
echo "=========================================="

# Query with tag filter
echo "Query filtered by category=electronics:"
$BENCH -h "$HOST" --cluster \
    -t vec-query \
    --dataset "$DATASET" \
    --search-prefix "$PREFIX" \
    --search-index "$INDEX_NAME" \
    --search-vector-field embedding \
    --tag-field category \
    --tag-filter "electronics" \
    -k 10 \
    -n 500 \
    -c 20 \
    --threads 4

echo ""
echo "=========================================="
echo "Distribution Statistics"
echo "=========================================="
echo ""
echo "Numeric field configurations used:"
echo "  price:      float:uniform:9.99:499.99:2"
echo "              - Uniform random between \$9.99 and \$499.99"
echo "              - 2 decimal places"
echo ""
echo "  quantity:   int:zipfian:1.5:1:1000"
echo "              - Zipfian distribution (skew=1.5)"
echo "              - Most values will be low (1-10)"
echo "              - Few values will be high (100-1000)"
echo ""
echo "  rating:     float:normal:4.0:0.5:1"
echo "              - Normal distribution, mean=4.0, stddev=0.5"
echo "              - Most values between 3.0 and 5.0"
echo "              - 1 decimal place"
echo ""
echo "  created_at: unix_timestamp:uniform:1672531200:1735689600"
echo "              - Unix timestamps from 2023-01-01 to 2024-12-31"
echo "              - Uniform distribution across 2 years"
echo ""
echo "Done!"
