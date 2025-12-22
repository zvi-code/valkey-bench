#!/bin/bash
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024-present, Zvi Schneider
#
# Unified Dataset Management - Shell Wrapper
#
# This file is part of valkey-search-benchmark and is licensed under the
# BSD 3-Clause License. See the LICENSE file in the root directory.
#
# Simple interface to the Python dataset manager
# Handles all downloads, conversions, and verifications
#
# Usage:
#   ./dataset.sh list                    # Show all available datasets
#   ./dataset.sh get sift-128            # Download and convert SIFT-128
#   ./dataset.sh get yfcc-10m            # Download YFCC with metadata
#   ./dataset.sh verify datasets/*.bin    # Verify all binaries
#   ./dataset.sh clean                   # Remove cache files
#

set -eo pipefail

# Find script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANAGER="${SCRIPT_DIR}/dataset_manager.py"

# Check for virtual environment and set Python command
VENV_DIR="${SCRIPT_DIR}/../venv"
if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/python3" ]; then
    PYTHON_CMD="$VENV_DIR/bin/python3"
else
    PYTHON_CMD="python3"
    echo -e "\033[1;33m⚠ Warning: Virtual environment not found. Using system python3.${NC}"
    echo -e "\033[1;33m  Run ./prereq-vectordbbench.sh first to set up the environment.${NC}"
    echo ""
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_usage() {
    cat << 'EOF'
Unified Dataset Manager

Commands:
  list [filter]              List available datasets (optionally filtered)
  get <dataset>              Download and convert dataset
  verify <file.bin>          Verify dataset integrity
  convert <args>             Convert custom dataset
  clean                      Remove cache and temporary files
  help                       Show this message

Examples:
  ./dataset.sh list
  ./dataset.sh list cohere
  ./dataset.sh get sift-128
  ./dataset.sh get yfcc-10m
  ./dataset.sh verify datasets/sift-128.bin
  ./dataset.sh clean

Available datasets include:
  - ANN-Benchmarks: sift-128, gist-960, glove-25/50/100, mnist, fashion-mnist, deep-96
  - BigANN: bigann-10m, deep-10m
  - BigANN+Metadata: yfcc-10m (200K tags for filtered search)
  - VectorDBBench: cohere-small-100k, cohere-medium-1m, cohere-large-10m
                   openai-medium-500k, openai-large-5m

For full list, run: ./dataset.sh list
EOF
}

# Check if Python script exists
if [ ! -f "$MANAGER" ]; then
    echo -e "${RED}✗ Dataset manager not found: $MANAGER${NC}"
    exit 1
fi

# Make sure it's executable
chmod +x "$MANAGER" 2>/dev/null || true

# Parse command
COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    list)
        # List datasets with optional filter
        if [ -n "$1" ]; then
            "$PYTHON_CMD" "$MANAGER" list --filter "$1"
        else
            "$PYTHON_CMD" "$MANAGER" list
        fi
        ;;
    
    get)
        # Download and convert dataset
        if [ -z "$1" ]; then
            echo -e "${RED}✗ Usage: $0 get <dataset-name>${NC}"
            echo "Example: $0 get sift-128"
            exit 1
        fi
        "$PYTHON_CMD" "$MANAGER" get "$@"
        ;;
    
    verify)
        # Verify one or more binaries
        if [ -z "$1" ]; then
            echo -e "${RED}✗ Usage: $0 verify <file.bin> [file2.bin ...]${NC}"
            exit 1
        fi
        
        FAILED=0
        for FILE in "$@"; do
            if [ ! -f "$FILE" ]; then
                echo -e "${RED}✗ File not found: $FILE${NC}"
                FAILED=1
                continue
            fi
            
            "$PYTHON_CMD" "$MANAGER" verify "$FILE" || FAILED=1
        done
        exit $FAILED
        ;;
    
    convert)
        # Custom conversion
        "$PYTHON_CMD" "$MANAGER" convert "$@"
        ;;
    
    clean)
        # Remove cache and temporary files
        echo "Cleaning cache and temporary files..."
        
        CACHE_DIR="/mnt/data/datasets/.cache"
        if [ -d "$CACHE_DIR" ]; then
            echo "  Removing cache: $CACHE_DIR"
            rm -rf "$CACHE_DIR"/*
        fi
        
        # Remove intermediate HDF5 files in download directory
        find /mnt/data/datasets -name "*.hdf5" -type f -exec rm -v {} \; 2>/dev/null || true
        
        echo -e "${GREEN}✓ Clean complete${NC}"
        ;;
    
    help|--help|-h)
        print_usage
        ;;
    
    *)
        echo -e "${RED}✗ Unknown command: $COMMAND${NC}"
        echo
        print_usage
        exit 1
        ;;
esac
