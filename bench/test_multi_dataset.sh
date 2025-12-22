#!/bin/bash
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024-present, Zvi Schneider
#
# Multi-dataset ef_search performance testing script
#
# This file is part of valkey-search-benchmark and is licensed under the
# BSD 3-Clause License. See the LICENSE file in the root directory.
#

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Environment variables with validation

# Benchmark project home (where datasets are stored)
BENCHMARK_HOME="${BENCHMARK_HOME:-/home/ubuntu/valkey-bench-rs}"
if [ ! -d "$BENCHMARK_HOME" ]; then
    echo "ERROR: BENCHMARK_HOME directory not found: $BENCHMARK_HOME"
    echo "Set BENCHMARK_HOME environment variable to benchmark project path"
    exit 1
fi

if [ -z "${HOST:-}" ]; then
    echo "ERROR: HOST environment variable not set"
    echo "Set HOST to the Valkey/Redis cluster endpoint"
    exit 1
fi

# Configuration
BINARY_DIR="${BENCHMARK_HOME}/datasets"
BENCHN="${BENCHMARK_HOME}/target/release/valkey-bench-rs"
# Use valkey-bench-rs --cli instead of valkey-cli
CLI="$BENCHN --cli"
CLUSTER_MODE=""

# Verify executables exist
if [ ! -x "$BENCHN" ]; then
    echo "ERROR: valkey-bench-rs not found or not executable: $BENCHN"
    echo "Run: cargo build --release"
    exit 1
fi

# Default test parameters
NUM_QUERIES=10000
CONCURRENCY=20
THREADS=20
EF_SEARCH_VALUES=(50 150 250 500)
K_NEIGHBORS=100  # Number of neighbors for ground truth

# Index limit (5 datasets max due to HNSW + FLAT indexes)
MAX_INDEXES=10  # Redis/Valkey limit
MAX_DATASETS=5  # Each dataset creates 2 indexes (HNSW + FLAT)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Dataset configurations
declare -A DATASET_CONFIG
# Format: "prefix,dimensions,expected_vectors,k_neighbors"
# Index name will be generated as: dataset-numvecs-dims-k

# === Standard ANN-Benchmarks ===
DATASET_CONFIG["glove-25"]="zvec_glove25:,25,1183514,100"
DATASET_CONFIG["glove-50"]="zvec_glove50:,50,1183514,100"
DATASET_CONFIG["glove-100"]="zvec_glove100:,100,1183514,100"
DATASET_CONFIG["sift-128"]="zvec_sift:,128,1000000,100"
DATASET_CONFIG["gist-960"]="zvec_gist:,960,1000000,100"
DATASET_CONFIG["fashion-mnist"]="zvec_fashion:,784,60000,100"
DATASET_CONFIG["deep-96"]="zvec_deep:,96,10000000,100"

# === Modern Production Datasets (Verified with proper ground truth) ===
DATASET_CONFIG["cohere-wiki-simple"]="zvec_cohere_wiki_simple:,768,486000,100"
DATASET_CONFIG["cohere-wiki-en"]="zvec_cohere_wiki_en:,768,35200000,100"
DATASET_CONFIG["fiqa"]="zvec_fiqa:,768,57638,100"
DATASET_CONFIG["msmarco"]="zvec_msmarco:,768,8800000,100"
DATASET_CONFIG["trec-covid"]="zvec_trec_covid:,768,171332,100"

# === VectorDB-Bench Datasets (Downloaded via vectordb-bench library) ===
DATASET_CONFIG["cohere-small-100k"]="zvec_cohere100k:,768,100000,100"
DATASET_CONFIG["cohere-medium-1m"]="zvec_cohere1m:,768,1000000,100"
DATASET_CONFIG["cohere-large-10m"]="zvec_cohere10m:,768,10000000,100"
DATASET_CONFIG["openai-medium-500k"]="zvec_openai500k:,1536,500000,100"
DATASET_CONFIG["openai-large-5m"]="zvec_openai5m:,1536,5000000,100"
DATASET_CONFIG["sift-medium-500k"]="zvec_sift500k:,128,500000,100"
DATASET_CONFIG["sift-large-5m"]="zvec_sift5m:,128,5000000,100"
DATASET_CONFIG["gist-small-100k"]="zvec_gist100k:,960,100000,100"
DATASET_CONFIG["gist-medium-1m"]="zvec_gist1m:,960,1000000,100"

# === Big-ANN Datasets (BIGANN format, billion-scale subsets) ===
DATASET_CONFIG["deep-10m"]="zvec_deep10m:,96,10000000,100"
DATASET_CONFIG["text2image-10m"]="zvec_t2i10m:,200,10000000,100"
DATASET_CONFIG["bigann-1m"]="zvec_bigann1m:,128,1000000,100"
DATASET_CONFIG["bigann-10m"]="zvec_bigann10m:,128,10000000,100"

# === Legacy datasets (for backward compatibility) ===
DATASET_CONFIG["openai-small"]="zvec_openai_small:,1536,50000,100"
DATASET_CONFIG["openai-medium"]="zvec_openai_medium:,1536,500000,100"
DATASET_CONFIG["dummy-test"]="zvec_dummy:dummy-test,8,1000,10"

# Function to generate index name
generate_index_name() {
    local dataset_name=$1
    IFS=',' read -r prefix dims expected_vectors k_neighbors <<< "${DATASET_CONFIG[$dataset_name]:-}"

    # Convert vectors to human-readable format (1M, 10M, etc)
    local vec_str=""
    if [ "$expected_vectors" -ge 1000000000 ]; then
        vec_str="$((expected_vectors / 1000000000))B"
    elif [ "$expected_vectors" -ge 1000000 ]; then
        vec_str="$((expected_vectors / 1000000))M"
    elif [ "$expected_vectors" -ge 1000 ]; then
        vec_str="$((expected_vectors / 1000))K"
    else
        vec_str="$expected_vectors"
    fi

    # Format: dataset-vectorcount-dimensions-k
    echo "${dataset_name}-${vec_str}-${dims}-${k_neighbors}"
}

# Function to cleanup old indexes to stay within limit
cleanup_old_indexes() {
    print_header "Checking index capacity"

    # Get list of current indexes
    local current_indexes=$($CLI -h "$HOST" -c --no-auth-warning FT._LIST 2>/dev/null | wc -l || echo "0")

    echo "Current indexes: $current_indexes / $MAX_INDEXES"

    if [ "$current_indexes" -ge "$MAX_INDEXES" ]; then
        print_warning "At index limit. Need to remove old indexes."

        # Get list of indexes
        local indexes=$($CLI -h "$HOST" -c --no-auth-warning FT._LIST 2>/dev/null || true)

        if [ -n "$indexes" ]; then
            echo "Current indexes:"
            echo "$indexes" | head -10

            read -p "Remove ALL existing indexes? [y/N] " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                for index in $indexes; do
                    echo "Dropping index: $index"
                    $CLI -h "$HOST" -c --no-auth-warning FT.DROPINDEX "$index" 2>/dev/null || true
                done
                print_success "Cleared all indexes"
            else
                print_error "Cannot proceed without clearing indexes (limit: $MAX_DATASETS datasets)"
                exit 1
            fi
        fi
    elif [ "$current_indexes" -gt "$((MAX_INDEXES - 2))" ]; then
        print_warning "Near index limit ($current_indexes/$MAX_INDEXES). Consider cleanup before next dataset."
    fi

    echo
}

# Function to test connection
test_connection() {
    echo -n "Testing connection to $HOST... "
    if timeout 5 $CLI -h "$HOST" -c --no-auth-warning PING > /dev/null 2>&1; then
        print_success "Connected"
        return 0
    else
        print_error "Connection failed to $HOST"
        exit 1
    fi
}

# Function to insert dataset vectors
insert_dataset() {
    local dataset_name=$1
    local binary_file="${BINARY_DIR}/${dataset_name}.bin"

    if [ ! -f "$binary_file" ]; then
        print_error "Dataset binary not found: $binary_file"
        echo "Run: ./download_and_prepare_datasets.sh --dataset $dataset_name"
        exit 1
    fi

    IFS=',' read -r prefix dims expected_vectors k_neighbors <<< "${DATASET_CONFIG[$dataset_name]:-}"

    # Generate proper index name
    local index_name=$(generate_index_name "$dataset_name")

    print_header "Inserting $dataset_name dataset"
    echo "  Index: $index_name"
    echo "  Dimensions: $dims"
    echo "  Expected vectors: $expected_vectors"
    echo "  K neighbors: $k_neighbors"
    echo

    # Check if we need to cleanup old indexes
    cleanup_old_indexes

    # Clear any existing data for this index
    echo "Clearing existing data for $index_name..."
    $CLI -h "$HOST" -c --no-auth-warning FT.DROPINDEX "$index_name" 2>/dev/null || true
    # Also clear the FLAT index
    $CLI -h "$HOST" -c --no-auth-warning FT.DROPINDEX "${index_name}_flat" 2>/dev/null || true

    # Insert dataset (ground truth phase)
    echo "Inserting vectors (this may take several minutes)..."
    local insertion_output=$(mktemp)
    echo "running command $BENCHN -h $HOST  --rfr no --dataset $binary_file -t vec-load --search --vector-dim $dims --search-name $index_name --search-prefix $prefix -n $expected_vectors -c 200"
    
    # Run without tee to allow progress bars to work correctly
    # Save output to file, but let progress bars display in real-time
    if ! $BENCHN -h "$HOST" --rfr no \
        --dataset "$binary_file" \
        -t vec-load --search --vector-dim "$dims" \
        --search-name "$index_name" --search-prefix "$prefix" \
        -n "$expected_vectors" -c 200 > "$insertion_output" 2>&1; then

        print_error "Failed to insert dataset"
        echo "Error output:"
        tail -20 "$insertion_output"
        rm -f "$insertion_output"
        exit 1
    fi

    # Display final status
    echo ""
    echo "Insertion completed. Final status:"
    tail -5 "$insertion_output"

    # Check for successful completion in output
    if ! grep -q "Dataset loaded" "$insertion_output"; then
        print_error "Dataset loading not confirmed in output"
        rm -f "$insertion_output"
        exit 1
    fi
    rm -f "$insertion_output"

    # Verify insertion with retries
    echo -n "Verifying insertion... "
    local retry_count=0
    local max_retries=5
    local num_docs=0

    while [ "$retry_count" -lt "$max_retries" ]; do
        sleep 2  # Wait for indexing to complete

        num_docs=$($CLI -h "$HOST" FT.INFO "$index_name" 2>/dev/null | \
            grep -A 1 'num_docs' | tail -1 | awk '{print $1}')

        if [ -n "$num_docs" ] && [ "$num_docs" -gt 0 ]; then
            break
        fi

        ((retry_count++))
        echo -n "."
    done

    if [ -z "$num_docs" ] || [ "$num_docs" -eq 0 ]; then
        print_error "Could not verify insertion - no documents found"
        exit 1
    fi

    echo " Indexed $num_docs documents"

    # Strict verification - must have at least 95% of expected vectors
    local min_acceptable=$((expected_vectors * 95 / 100))
    if [ "$num_docs" -lt "$min_acceptable" ]; then
        print_error "Insufficient vectors indexed: $num_docs/$expected_vectors (minimum: $min_acceptable)"
        exit 1
    fi

    print_success "Dataset inserted successfully: $num_docs vectors"
    echo
    return 0
}

# Function to run ef_search tests for a dataset
run_ef_search_tests() {
    local dataset_name=$1
    local binary_file="${BINARY_DIR}/${dataset_name}.bin"

    if [ ! -f "$binary_file" ]; then
        print_error "Dataset binary not found: $binary_file"
        exit 1
    fi

    IFS=',' read -r prefix dims expected_vectors k_neighbors <<< "${DATASET_CONFIG[$dataset_name]:-}"

    # Generate proper index name
    local index_name=$(generate_index_name "$dataset_name")
    local results_file="${RESULTS_DIR}/ef_search_results_${dataset_name}_$(date +%Y%m%d_%H%M%S).csv"

    print_header "Testing ef_search performance for $dataset_name"
    echo "  Index: $index_name"
    echo "  Queries: $NUM_QUERIES"
    echo "  ef_search values: ${EF_SEARCH_VALUES[*]}"
    echo

    # Create CSV header
    echo "dataset,ef_search,avg_recall,min_recall,max_recall,queries,qps,latency_ms,timestamp" > "$results_file"

    # Table header
    echo "╔═══════════╦════════════╦════════════╦════════════╦═════════╦═══════════╗"
    echo "║ ef_search ║ Avg Recall ║ Min Recall ║ Max Recall ║   QPS   ║  Latency  ║"
    echo "╠═══════════╬════════════╬════════════╬════════════╬═════════╬═══════════╣"

    local test_failed=false
    for ef_search in "${EF_SEARCH_VALUES[@]}"; do
        # Run benchmark with ef_search parameter
        local benchmark_output=$(mktemp)
        if ! $BENCHN -h "$HOST" --rfr no \
            --dataset "$binary_file" \
            -t vec-query --search --vector-dim "$dims" \
            --search-name "$index_name" --search-prefix "$prefix" \
            --ef-search "$ef_search" \
            -n "$NUM_QUERIES" -c "$CONCURRENCY" --threads "$THREADS" --nocontent > "$benchmark_output" 2>&1; then

            print_error "Benchmark failed for ef_search=$ef_search"
            echo "Error output:"
            tail -20 "$benchmark_output"
            rm -f "$benchmark_output"
            test_failed=true
            break
        fi

        OUTPUT=$(cat "$benchmark_output")
        rm -f "$benchmark_output"

        # Extract statistics
        AVG_RECALL=$(echo "$OUTPUT" | grep "Average:" | awk '{print $2}' | sed 's/%//')
        MIN_RECALL=$(echo "$OUTPUT" | grep "Min:" | awk '{print $2}' | sed 's/%//')
        MAX_RECALL=$(echo "$OUTPUT" | grep "Max:" | awk '{print $2}' | sed 's/%//')
        TOTAL_QUERIES=$(echo "$OUTPUT" | grep "Queries evaluated:" | awk '{print $3}')
        QPS=$(echo "$OUTPUT" | grep "throughput summary:" | awk '{print $3}')
        LATENCY=$(echo "$OUTPUT" | grep "avg       min" -A 1 | tail -1 | awk '{print $1}')

        # Check if we got valid results
        if [ -z "$AVG_RECALL" ] || [ "$AVG_RECALL" = "N/A" ]; then
            print_error "No recall data found for ef_search=$ef_search"
            test_failed=true
            break
        fi

        # Default values if not found
        AVG_RECALL=${AVG_RECALL:-"N/A"}
        MIN_RECALL=${MIN_RECALL:-"N/A"}
        MAX_RECALL=${MAX_RECALL:-"N/A"}
        TOTAL_QUERIES=${TOTAL_QUERIES:-"0"}
        QPS=${QPS:-"N/A"}
        LATENCY=${LATENCY:-"N/A"}

        # Save to CSV
        echo "$dataset_name,$ef_search,$AVG_RECALL,$MIN_RECALL,$MAX_RECALL,$TOTAL_QUERIES,$QPS,$LATENCY,$(date +%Y-%m-%d_%H:%M:%S)" >> "$results_file"

        # Display in table - print complete row at once
        printf "║ %-9s ║ %9s%% ║ %9s%% ║ %9s%% ║ %7s ║ %9s ║\n" \
            "$ef_search" "$AVG_RECALL" "$MIN_RECALL" "$MAX_RECALL" "$QPS" "${LATENCY}ms"
    done

    echo "╚═══════════╩════════════╩════════════╩════════════╩═════════╩═══════════╝"
    echo

    # Check if tests failed
    if [ "$test_failed" = true ]; then
        print_error "Test suite failed for $dataset_name"
        exit 1
    fi

    # Analysis
    echo "=== ANALYSIS for $dataset_name ==="

    # Find optimal configurations
    BEST_RECALL=$(tail -n +2 "$results_file" | sort -t',' -k3 -nr | head -1)
    BEST_QPS=$(tail -n +2 "$results_file" | sort -t',' -k7 -nr | head -1)

    if [ -n "$BEST_RECALL" ]; then
        BEST_RECALL_EF=$(echo "$BEST_RECALL" | cut -d',' -f2)
        BEST_RECALL_VALUE=$(echo "$BEST_RECALL" | cut -d',' -f3)
        echo "  Highest recall: ${BEST_RECALL_VALUE}% at ef_search=$BEST_RECALL_EF"
    fi

    if [ -n "$BEST_QPS" ]; then
        BEST_QPS_EF=$(echo "$BEST_QPS" | cut -d',' -f2)
        BEST_QPS_VALUE=$(echo "$BEST_QPS" | cut -d',' -f7)
        echo "  Highest QPS: ${BEST_QPS_VALUE} at ef_search=$BEST_QPS_EF"
    fi

    echo
    echo "Results saved to: $results_file"
    echo

    return 0
}

# Parse command line arguments
SELECTED_DATASETS=()
INSERT_ONLY=false
TEST_ONLY=false
SKIP_INSERT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            SELECTED_DATASETS+=("$2")
            shift 2
            ;;
        --all)
            # Test all available datasets
            for dataset in "${!DATASET_CONFIG[@]}"; do
                binary_file="${BINARY_DIR}/${dataset}.bin"
                if [ -f "$binary_file" ]; then
                    SELECTED_DATASETS+=("$dataset")
                fi
            done
            shift
            ;;
        --insert-only)
            INSERT_ONLY=true
            shift
            ;;
        --test-only)
            TEST_ONLY=true
            SKIP_INSERT=true
            shift
            ;;
        --skip-insert)
            SKIP_INSERT=true
            shift
            ;;
        --queries)
            NUM_QUERIES=$2
            shift 2
            ;;
        --ef-search)
            IFS=',' read -ra EF_SEARCH_VALUES <<< "$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --dataset NAME       Test specific dataset"
            echo "  --all                Test all available datasets"
            echo "  --insert-only        Only insert datasets, don't run tests"
            echo "  --test-only          Only run tests, skip insertion"
            echo "  --skip-insert        Skip insertion if index exists"
            echo "  --queries N          Number of queries per test (default: 1000)"
            echo "  --ef-search VALUES   Comma-separated ef_search values"
            echo "  --help               Show this help message"
            echo ""
            echo "Available datasets:"
            for key in "${!DATASET_CONFIG[@]}"; do
                IFS=',' read -r prefix dims vectors k_neighbors <<< "${DATASET_CONFIG[$key]}"
                index_name=$(generate_index_name "$key")
                printf "  %-15s %4s dims, %10s vectors (index: %s)\n" "$key" "$dims" "$vectors" "$index_name"
            done | sort
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default to cohere-small-100k if no dataset specified
if [ ${#SELECTED_DATASETS[@]} -eq 0 ]; then
    SELECTED_DATASETS=("cohere-small-100k")
fi

# Main execution
print_header "Multi-Dataset ef_search Performance Test"
echo "Host: $HOST"
echo "Datasets: ${SELECTED_DATASETS[*]}"
echo

# Test connection
test_connection || exit 1
echo

# Create results directory
RESULTS_DIR="${BENCHMARK_HOME}/ef_search_results"
mkdir -p "$RESULTS_DIR"

# Process each dataset
for dataset_name in "${SELECTED_DATASETS[@]}"; do
    if [ -z "${DATASET_CONFIG[$dataset_name]:-}" ]; then
        print_error "Unknown dataset: $dataset_name"
        echo "Available datasets: ${!DATASET_CONFIG[*]}"
        exit 1
    fi

    # Insert dataset if needed
    if [ "$TEST_ONLY" = false ] && [ "$SKIP_INSERT" = false ]; then
        insert_dataset "$dataset_name"  # Will exit on failure now
    fi

    # Run tests if needed
    if [ "$INSERT_ONLY" = false ]; then
        run_ef_search_tests "$dataset_name"  # Will exit on failure now
    fi
done

# Generate comparison report if multiple datasets were tested
if [ ${#SELECTED_DATASETS[@]} -gt 1 ] && [ "$INSERT_ONLY" = false ]; then
    print_header "Cross-Dataset Comparison"

    COMPARISON_FILE="${RESULTS_DIR}/comparison_$(date +%Y%m%d_%H%M%S).csv"
    echo "dataset,dimensions,vectors,best_recall,best_recall_ef,best_qps,best_qps_ef" > "$COMPARISON_FILE"

    for dataset_name in "${SELECTED_DATASETS[@]}"; do
        # Find the most recent results file for this dataset
        LATEST_RESULTS=$(ls -t "${RESULTS_DIR}"/ef_search_results_${dataset_name}_*.csv 2>/dev/null | head -1)

        if [ -f "$LATEST_RESULTS" ]; then
            IFS=',' read -r prefix dims vectors k_neighbors <<< "${DATASET_CONFIG[$dataset_name]:-}"

            # Get best recall and QPS
            BEST_RECALL_LINE=$(tail -n +2 "$LATEST_RESULTS" | sort -t',' -k3 -nr | head -1)
            BEST_QPS_LINE=$(tail -n +2 "$LATEST_RESULTS" | sort -t',' -k7 -nr | head -1)

            BEST_RECALL=$(echo "$BEST_RECALL_LINE" | cut -d',' -f3)
            BEST_RECALL_EF=$(echo "$BEST_RECALL_LINE" | cut -d',' -f2)
            BEST_QPS=$(echo "$BEST_QPS_LINE" | cut -d',' -f7)
            BEST_QPS_EF=$(echo "$BEST_QPS_LINE" | cut -d',' -f2)

            echo "$dataset_name,$dims,$vectors,$BEST_RECALL,$BEST_RECALL_EF,$BEST_QPS,$BEST_QPS_EF" >> "$COMPARISON_FILE"

            printf "  %-15s: %4sd, Best Recall: %s%% (ef=%s), Best QPS: %s (ef=%s)\n" \
                "$dataset_name" "$dims" "$BEST_RECALL" "$BEST_RECALL_EF" "$BEST_QPS" "$BEST_QPS_EF"
        fi
    done

    echo
    echo "Comparison saved to: $COMPARISON_FILE"
fi

print_success "Testing complete!"