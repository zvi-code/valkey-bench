#!/bin/bash
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024-present, Zvi Schneider
#
# Stage Monitor - Example external monitoring script
#
# This script demonstrates how to monitor [STAGE:START/END] signals from
# benchmark wrappers and trigger external profiling tools during specific
# test stages.
#
# Usage:
#   Terminal 1: Run benchmark wrapper
#     ./max_qps_recall.py --host localhost --dataset data.bin ... 2>&1 | tee bench.log
#
#   Terminal 2: Monitor and collect perf
#     ./stage-monitor.sh bench.log --collect-perf --watch vec-query
#
# Or pipe directly:
#     ./max_qps_recall.py ... 2>&1 | ./stage-monitor.sh --collect-perf --watch vec-query
#

set -euo pipefail

# Configuration
WATCH_STAGES=()           # Stages to monitor (empty = all stages)
COLLECT_PERF=false        # Collect perf stats
PERF_SCRIPT=""            # Path to perf collection script
SERVER_PID=""             # Server PID for perf collection
OUTPUT_DIR="stage_perf"   # Output directory for perf data
PERF_PID=""               # Background perf process PID

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [INPUT_FILE]

Monitor [STAGE:START/END] signals and optionally trigger perf collection.

OPTIONS:
    --watch STAGE       Monitor specific stage (can be specified multiple times)
                        e.g., --watch vec-query --watch vec-load
                        If not specified, monitors all stages
    
    --collect-perf      Enable perf data collection during monitored stages
    
    --perf-script PATH  Path to perf collection script
                        (default: ../perf/collect-stats.sh)
    
    --server-pid PID    Server process ID for perf collection
                        (default: auto-detect valkey-server)
    
    --output-dir DIR    Output directory for perf data
                        (default: ./stage_perf)
    
    -h, --help          Show this help message

INPUT_FILE:
    File to monitor (optional, reads from stdin if not specified)

EXAMPLES:
    # Monitor all stages from stdin
    ./max_qps_recall.py ... 2>&1 | $0
    
    # Monitor specific stages with perf collection
    ./max_qps_recall.py ... 2>&1 | $0 --watch vec-query --collect-perf
    
    # Monitor log file
    tail -f bench.log | $0 --watch vec-query --collect-perf --server-pid 12345
    
    # With custom perf script
    $0 --collect-perf --perf-script /path/to/my-perf-collector.sh < bench.log

STAGE SIGNAL FORMAT:
    [STAGE:START] stage_name[:tag]
    [STAGE:END] stage_name[:tag] duration=X.XXXs

EOF
}

# Parse arguments
INPUT_FILE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --watch)
            WATCH_STAGES+=("$2")
            shift 2
            ;;
        --collect-perf)
            COLLECT_PERF=true
            shift
            ;;
        --perf-script)
            PERF_SCRIPT="$2"
            shift 2
            ;;
        --server-pid)
            SERVER_PID="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            INPUT_FILE="$1"
            shift
            ;;
    esac
done

# Setup perf collection if requested
if [ "$COLLECT_PERF" = true ]; then
    # Find perf script
    if [ -z "$PERF_SCRIPT" ]; then
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        PERF_SCRIPT="${SCRIPT_DIR}/../perf/collect-stats.sh"
    fi
    
    if [ ! -x "$PERF_SCRIPT" ]; then
        echo -e "${RED}Error: Perf script not found or not executable: $PERF_SCRIPT${NC}" >&2
        exit 1
    fi
    
    # Find server PID if not specified
    if [ -z "$SERVER_PID" ]; then
        SERVER_PID=$(pgrep -f "valkey-server|redis-server" | head -1)
        if [ -z "$SERVER_PID" ]; then
            echo -e "${RED}Error: Could not auto-detect server PID. Use --server-pid${NC}" >&2
            exit 1
        fi
        echo -e "${BLUE}Auto-detected server PID: $SERVER_PID${NC}"
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    echo -e "${BLUE}Perf data will be saved to: $OUTPUT_DIR${NC}"
fi

# Cleanup function
cleanup() {
    if [ -n "$PERF_PID" ] && kill -0 "$PERF_PID" 2>/dev/null; then
        echo -e "\n${YELLOW}Stopping perf collection (PID: $PERF_PID)${NC}"
        kill "$PERF_PID" 2>/dev/null || true
        wait "$PERF_PID" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

# Check if stage should be monitored
should_monitor_stage() {
    local stage="$1"
    
    # If no watch list, monitor all stages
    if [ ${#WATCH_STAGES[@]} -eq 0 ]; then
        return 0
    fi
    
    # Check if stage matches any watch pattern
    for watch_stage in "${WATCH_STAGES[@]}"; do
        if [[ "$stage" == "$watch_stage"* ]]; then
            return 0
        fi
    done
    
    return 1
}

# Start perf collection
start_perf() {
    local stage="$1"
    local timestamp="$(date +%Y%m%d_%H%M%S)"
    local output_file="${OUTPUT_DIR}/perf_${stage}_${timestamp}.log"
    
    echo -e "${GREEN}► Starting perf collection for stage: $stage${NC}"
    echo -e "${BLUE}  Output: $output_file${NC}"
    
    # Start perf in background
    "$PERF_SCRIPT" "$SERVER_PID" > "$output_file" 2>&1 &
    PERF_PID=$!
    
    echo -e "${BLUE}  Perf PID: $PERF_PID${NC}"
}

# Stop perf collection
stop_perf() {
    local stage="$1"
    local duration="$2"
    
    if [ -n "$PERF_PID" ] && kill -0 "$PERF_PID" 2>/dev/null; then
        echo -e "${GREEN}◼ Stopping perf collection for stage: $stage (duration: $duration)${NC}"
        kill "$PERF_PID" 2>/dev/null || true
        wait "$PERF_PID" 2>/dev/null || true
        PERF_PID=""
    fi
}

# Main monitoring loop
monitor_stages() {
    local input_source="$1"
    
    echo -e "${BLUE}=== Stage Monitor Active ===${NC}"
    if [ ${#WATCH_STAGES[@]} -gt 0 ]; then
        echo -e "${BLUE}Watching stages: ${WATCH_STAGES[*]}${NC}"
    else
        echo -e "${BLUE}Monitoring all stages${NC}"
    fi
    echo -e "${BLUE}===========================${NC}\n"
    
    while IFS= read -r line; do
        # Always echo the line (pass through)
        echo "$line"
        
        # Check for stage signals
        if [[ "$line" =~ \[STAGE:START\]\ ([^ ]+) ]]; then
            stage="${BASH_REMATCH[1]}"
            
            if should_monitor_stage "$stage"; then
                echo -e "${YELLOW}⚡ Stage started: $stage${NC}"
                
                if [ "$COLLECT_PERF" = true ]; then
                    start_perf "$stage"
                fi
            fi
            
        elif [[ "$line" =~ \[STAGE:END\]\ ([^ ]+)\ duration=([0-9.]+)s ]]; then
            stage="${BASH_REMATCH[1]}"
            duration="${BASH_REMATCH[2]}"
            
            if should_monitor_stage "$stage"; then
                echo -e "${YELLOW}✓ Stage completed: $stage (${duration}s)${NC}"
                
                if [ "$COLLECT_PERF" = true ]; then
                    stop_perf "$stage" "$duration"
                fi
            fi
        fi
    done < "$input_source"
}

# Run monitor
if [ -n "$INPUT_FILE" ]; then
    if [ ! -f "$INPUT_FILE" ]; then
        echo -e "${RED}Error: Input file not found: $INPUT_FILE${NC}" >&2
        exit 1
    fi
    monitor_stages "$INPUT_FILE"
else
    # Read from stdin
    monitor_stages /dev/stdin
fi

echo -e "\n${BLUE}=== Stage Monitor Stopped ===${NC}"
