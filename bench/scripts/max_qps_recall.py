#!/usr/bin/env python3
"""
SPDX-License-Identifier: BSD-3-Clause

Copyright (c) 2024-present, Zvi Schneider

Find maximum QPS at target recall threshold.

This script implements TODO Wrapper #1: automated search for the maximum
achievable QPS while maintaining a specified recall threshold.

The script uses binary search to efficiently find the optimal number of clients
(or other parameter) that maximizes throughput while ensuring recall meets
the target threshold.

Example usage:
    # Find max QPS for recall >= 95% with ef_search=100
    ./max_qps_recall.py \\
        --host localhost \\
        --dataset datasets/sift-128.bin \\
        --target-recall 0.95 \\
        --ef-search 100 \\
        --output results.csv
    
    # With external monitoring (in another terminal):
    ./stage-monitor.sh --watch vec-query --collect-perf
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wrappers import (
    ValKeyBenchmarkWrapper,
    BenchmarkConfig,
    BenchmarkError,
    BinaryNotFoundError,
)


def main():
    parser = argparse.ArgumentParser(
        description="Find maximum QPS at target recall threshold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s --host localhost --dataset sift-128.bin --target-recall 0.95
  
  # With specific ef_search and output file
  %(prog)s --host localhost --dataset sift-128.bin \\
      --target-recall 0.95 --ef-search 100 --output results.csv
  
  # Search over ef_search instead of num_clients
  %(prog)s --host localhost --dataset sift-128.bin \\
      --target-recall 0.95 --search-param ef_search \\
      --min-value 50 --max-value 500
  
  # With latency constraint
  %(prog)s --host localhost --dataset sift-128.bin \\
      --target-recall 0.95 --max-p99-latency 10.0

Notes:
  - The script emits [STAGE:START/END] signals to stderr for external monitoring
  - Use stage-monitor.sh to collect perf data during vec-query stages
  - Results are saved to CSV if --output is specified
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--host",
        required=True,
        help="Redis/Valkey server hostname or IP"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to binary dataset file"
    )
    
    # Search parameters
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="Target recall threshold (0.0-1.0, default: 0.95)"
    )
    parser.add_argument(
        "--search-param",
        choices=["num_clients", "ef_search", "num_threads"],
        default="num_clients",
        help="Parameter to search over (default: num_clients)"
    )
    parser.add_argument(
        "--min-value",
        type=int,
        default=1,
        help="Minimum parameter value (default: 1)"
    )
    parser.add_argument(
        "--max-value",
        type=int,
        default=200,
        help="Maximum parameter value (default: 200)"
    )
    
    # Benchmark configuration
    parser.add_argument(
        "--ef-search",
        type=int,
        help="HNSW ef_search parameter (required if not searching over it)"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=10,
        help="Number of clients (default: 10, used if not search param)"
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads (default: 4, used if not search param)"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10000,
        help="Number of requests (default: 10000)"
    )
    parser.add_argument(
        "--operation",
        default="vec-query",
        help="Operation type (default: vec-query)"
    )
    
    # Optional constraints
    parser.add_argument(
        "--max-p99-latency",
        type=float,
        help="Maximum P99 latency constraint in milliseconds"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        help="Output CSV file for results"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.target_recall < 0.0 or args.target_recall > 1.0:
        parser.error("--target-recall must be between 0.0 and 1.0")
    
    if args.search_param != "ef_search" and args.ef_search is None:
        parser.error("--ef-search is required when not searching over it")
    
    if not Path(args.dataset).exists():
        parser.error(f"Dataset file not found: {args.dataset}")
    
    # Initialize wrapper
    try:
        wrapper = ValKeyBenchmarkWrapper(verbose=args.verbose)
    except BinaryNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    # Create base configuration
    base_config = BenchmarkConfig(
        host=args.host,
        dataset=args.dataset,
        num_clients=args.num_clients,
        num_threads=args.num_threads,
        num_requests=args.num_requests,
        ef_search=args.ef_search,
        operation=args.operation,
    )
    
    print(f"\n{'='*70}")
    print(f"Finding Maximum QPS at Target Recall")
    print(f"{'='*70}")
    print(f"Host:          {args.host}")
    print(f"Dataset:       {args.dataset}")
    print(f"Operation:     {args.operation}")
    print(f"Target Recall: {args.target_recall:.2%}")
    if args.max_p99_latency:
        print(f"Max P99:       {args.max_p99_latency:.3f} ms")
    print(f"Search Param:  {args.search_param} [{args.min_value}, {args.max_value}]")
    print(f"{'='*70}\n")
    
    # Run search
    try:
        if args.max_p99_latency:
            # Search with latency constraint
            best_result = wrapper.find_max_qps_with_constraints(
                base_config=base_config,
                min_recall=args.target_recall,
                max_latency_p99=args.max_p99_latency,
                param_name=args.search_param,
                min_val=args.min_value,
                max_val=args.max_value,
            )
        else:
            # Simple binary search for max QPS
            best_result = wrapper.binary_search_max_qps(
                base_config=base_config,
                target_recall=args.target_recall,
                param_name=args.search_param,
                min_val=args.min_value,
                max_val=args.max_value,
            )
        
        # Print results
        print(f"\n{'='*70}")
        if best_result:
            print("✓ OPTIMAL CONFIGURATION FOUND")
            print(f"{'='*70}")
            print(f"\nParameter Value: {args.search_param} = "
                  f"{getattr(best_result.config, args.search_param)}")
            print(f"\nPerformance:")
            print(f"  QPS:         {best_result.qps:.1f} requests/sec")
            print(f"  Latency Avg: {best_result.latency_avg:.3f} ms")
            print(f"  Latency P50: {best_result.latency_p50:.3f} ms")
            print(f"  Latency P95: {best_result.latency_p95:.3f} ms")
            print(f"  Latency P99: {best_result.latency_p99:.3f} ms")
            
            if best_result.recall_avg is not None:
                print(f"\nRecall:")
                print(f"  Average:     {best_result.recall_avg:.2%}")
                print(f"  Min:         {best_result.recall_min:.2%}")
                print(f"  Max:         {best_result.recall_max:.2%}")
            
            if best_result.baseline_latency_avg is not None:
                overhead = best_result.latency_avg - best_result.baseline_latency_avg
                print(f"\nBaseline:")
                print(f"  Network RTT: {best_result.baseline_latency_avg:.3f} ms")
                print(f"  Overhead:    {overhead:.3f} ms")
            
            # Save to CSV if requested
            if args.output:
                wrapper.save_results_csv([best_result], args.output)
                print(f"\nResults saved to: {args.output}")
            
            print(f"{'='*70}\n")
            return 0
        else:
            print("✗ COULD NOT ACHIEVE TARGET")
            print(f"{'='*70}")
            print(f"\nTarget recall of {args.target_recall:.2%} could not be achieved")
            print(f"within the parameter range [{args.min_value}, {args.max_value}]")
            print(f"\nSuggestions:")
            print(f"  - Increase --max-value")
            if args.search_param == "num_clients":
                print(f"  - Try increasing --ef-search")
            elif args.search_param == "ef_search":
                print(f"  - Try increasing --num-clients")
            print(f"  - Lower --target-recall threshold")
            print(f"{'='*70}\n")
            return 1
    
    except BenchmarkError as e:
        print(f"\nError during benchmark: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
