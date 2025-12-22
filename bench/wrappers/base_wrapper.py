#!/usr/bin/env python3
"""
SPDX-License-Identifier: BSD-3-Clause

Copyright (c) 2024-present, Zvi Schneider

Simple, elegant wrapper framework for valkey-bench-rs.

This module provides a clean Python interface for building sophisticated
benchmark wrappers using the Keep It Simple principle. The Rust binary handles
cluster detection and auto-optimization - the wrapper orchestrates and parses results.

Example:
    wrapper = ValKeyBenchmarkWrapper(verbose=True)
    config = BenchmarkConfig(host="localhost", dataset="sift-128.bin")

    with wrapper.stage("vec-query", tag="ef_100"):
        result = wrapper.run(config, operation="vec-query", ef_search=100)

    print(f"QPS: {result.qps:.1f}, Recall: {result.recall_avg:.2%}")
"""

from dataclasses import dataclass, field, replace
from typing import List, Optional, Dict, Any, Callable, Tuple
from pathlib import Path
from contextlib import contextmanager
import subprocess
import time
import re
import sys
import os
import csv as csv_module
import struct


# ============================================================================
# Custom Exceptions
# ============================================================================

class BenchmarkError(Exception):
    """Base exception for benchmark errors."""
    pass


class BinaryNotFoundError(BenchmarkError):
    """Raised when valkey-benchmark or valkey-cli binary cannot be found."""
    pass


class ParseError(BenchmarkError):
    """Raised when output parsing fails."""
    pass


# ============================================================================
# Dataset Utilities
# ============================================================================

def find_dataset_path(dataset_name: str) -> str:
    """Find dataset file path from dataset name.
    
    Looks for dataset in standard locations:
    1. BENCHMARK_HOME/datasets/ if BENCHMARK_HOME is set
    2. datasets/ directory (relative to project root)
    3. Current directory
    
    Args:
        dataset_name: Dataset name (e.g., 'openai-large-5m' or 'sift-128.bin')
    
    Returns:
        Resolved absolute path to dataset file
    
    Raises:
        BenchmarkError: If dataset file cannot be found
    """
    search_paths = []
    
    # Add standard locations
    benchmark_home = os.environ.get('BENCHMARK_HOME')
    if benchmark_home:
        search_paths.append(Path(benchmark_home) / "datasets")
    
    # Relative to project root (3 levels up from base_wrapper.py)
    project_root = Path(__file__).parent.parent.parent
    search_paths.append(project_root / "datasets")
    
    # Current directory
    search_paths.append(Path.cwd())
    
    # Try with .bin extension if not provided
    if not dataset_name.endswith('.bin'):
        dataset_name = f"{dataset_name}.bin"
    
    for search_path in search_paths:
        candidate = search_path / dataset_name
        if candidate.exists():
            return str(candidate.resolve())
    
    raise BenchmarkError(
        f"Dataset '{dataset_name}' not found in:\n" + 
        "\n".join(f"  - {p}" for p in search_paths)
    )


def detect_dataset_info(dataset_path: str) -> Tuple[int, int]:
    """Read dataset binary header to get dimensions and vector count.
    
    Binary format (first 4KB header):
        0-3:   Magic number (0xDECDB001)
        4-7:   Version (uint32)
        8-263: Dataset name (char[256])
        264-267: distance_metric, dtype, has_metadata, padding (4 bytes)
        268-271: Dimensions (uint32)
        272-279: Number of vectors (uint64)
        280-287: Number of queries (uint64)
    
    Args:
        dataset_path: Path to binary dataset file
    
    Returns:
        Tuple of (dimensions, num_vectors)
    
    Raises:
        BenchmarkError: If dataset has invalid format or magic number
        IOError: If dataset file cannot be read
    """
    with open(dataset_path, 'rb') as f:
        # Read header (4KB)
        header = f.read(4096)
        
        if len(header) < 288:
            raise BenchmarkError(
                f"Dataset file too small: {len(header)} bytes "
                f"(expected at least 288 bytes for header)"
            )
        
        # Parse header fields at correct offsets
        magic, version = struct.unpack('<II', header[0:8])
        
        # Validate magic number
        expected_magic = 0xDECDB001
        if magic != expected_magic:
            raise BenchmarkError(
                f"Invalid dataset magic number: 0x{magic:08X} "
                f"(expected 0x{expected_magic:08X})"
            )
        
        # Skip dataset_name (256 bytes) and flags (4 bytes) to get to dimensions
        dimensions = struct.unpack('<I', header[268:272])[0]
        num_vectors = struct.unpack('<Q', header[272:280])[0]
        
        return dimensions, num_vectors


def generate_index_name(dataset_name: str, num_vectors: int, dimensions: int, k: int = 100) -> str:
    """Generate index name following standard naming convention.
    
    Format: {dataset}-{vector_count}-{dimensions}-{k}
    Examples:
        - openai-large-5m-5M-1536-100
        - sift-128-1M-128-100
    
    Args:
        dataset_name: Base dataset name (with or without .bin extension)
        num_vectors: Number of vectors in dataset
        dimensions: Vector dimensionality
        k: Number of neighbors (default: 100)
    
    Returns:
        Formatted index name string
    """
    # Remove .bin extension if present
    base_name = Path(dataset_name).stem
    
    # Format vector count: 5000000 -> "5M", 100000 -> "100K"
    if num_vectors >= 1_000_000:
        vec_str = f"{num_vectors // 1_000_000}M"
    elif num_vectors >= 1_000:
        vec_str = f"{num_vectors // 1_000}K"
    else:
        vec_str = str(num_vectors)
    
    # Index name format: dataset-{M/K}-{dimensions}-{k}
    return f"{base_name}-{vec_str}-{dimensions}-{k}"


def generate_search_prefix(dataset_name: str) -> str:
    """Generate search key prefix following standard naming convention.
    
    Format: zvec_{shortname}:
    Examples:
        - openai-large-5m -> zvec_openai5m:
        - openai-medium-500k -> zvec_openai500k
        - sift-128 -> zvec_sift128
        - cohere-large-10m -> zvec_cohere10m
    
    The short name removes size qualifiers (large/medium/small) and dashes.
    This matches the actual key format in the cluster.
    
    Args:
        dataset_name: Base dataset name (with or without .bin extension)
    
    Returns:
        Formatted search prefix string (WITHOUT trailing colon)
    """
    # Remove .bin extension if present
    base_name = Path(dataset_name).stem
    
    # Remove size qualifiers (large, medium, small) and their surrounding dashes
    short_name = base_name.lower()
    for size_word in ['-large-', '-medium-', '-small-', '-wiki-']:
        short_name = short_name.replace(size_word, '-')
    
    # Remove all dashes and underscores for short name
    short_name = short_name.replace('-', '').replace('_', '')
    
    # Search prefix format: zvec_{shortname} (no trailing colon)
    return f"zvec_{short_name}:"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.
    
    Attributes:
        host: Redis/Valkey server hostname or IP
        dataset: Path to binary dataset file (optional)
        num_clients: Number of parallel clients (default: 10)
        num_threads: Number of threads (default: 4)
        num_requests: Number of requests to execute (default: 10000)
        ef_search: HNSW ef_search parameter (optional)
        operation: Operation type for -t flag (default: "vec-query")
        extra_args: Additional CLI arguments as list (appended at end)
        prefix_args: Arguments to insert before -t flag (e.g., optimizer flags)
    """
    host: str
    dataset: Optional[str] = None
    num_clients: int = 10
    num_threads: int = 4
    num_requests: int = 10000
    ef_search: Optional[int] = None
    operation: str = "vec-query"
    extra_args: List[str] = field(default_factory=list)
    prefix_args: List[str] = field(default_factory=list)
    
    def to_args(self) -> List[str]:
        """Convert configuration to CLI arguments.
        
        Returns:
            List of command-line arguments
        """
        args = [
            "-h", self.host,
            "-c", str(self.num_clients),
            "--threads", str(self.num_threads),
            "-n", str(self.num_requests),
        ]
        
        if self.dataset:
            args.extend(["--dataset", self.dataset])
        
        if self.ef_search is not None:
            args.extend(["--ef-search", str(self.ef_search)])
        
        args.extend(self.extra_args)
        return args


@dataclass
class BenchmarkResult:
    """Results from a benchmark run.
    
    Attributes:
        qps: Queries per second (throughput)
        latency_avg: Average latency in milliseconds
        latency_p50: P50 latency in milliseconds
        latency_p95: P95 latency in milliseconds
        latency_p99: P99 latency in milliseconds
        latency_max: Maximum latency in milliseconds
        recall_avg: Average recall (0.0-1.0), None if not applicable
        recall_min: Minimum recall (0.0-1.0), None if not applicable
        recall_max: Maximum recall (0.0-1.0), None if not applicable
        baseline_latency_avg: Baseline network latency avg, None if not measured
        config: The configuration used for this run
        timestamp: Unix timestamp when benchmark completed
    """
    qps: float
    latency_avg: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    latency_max: float
    recall_avg: Optional[float] = None
    recall_min: Optional[float] = None
    recall_max: Optional[float] = None
    baseline_latency_avg: Optional[float] = None
    config: Optional[BenchmarkConfig] = None
    timestamp: float = field(default_factory=time.time)
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [
            f"QPS: {self.qps:.1f}",
            f"Latency: avg={self.latency_avg:.3f}ms, p50={self.latency_p50:.3f}ms, "
            f"p95={self.latency_p95:.3f}ms, p99={self.latency_p99:.3f}ms",
        ]
        if self.recall_avg is not None:
            lines.append(
                f"Recall: avg={self.recall_avg:.2%}, "
                f"min={self.recall_min:.2%}, max={self.recall_max:.2%}"
            )
        if self.baseline_latency_avg is not None:
            overhead = self.latency_avg - self.baseline_latency_avg
            lines.append(
                f"Baseline: {self.baseline_latency_avg:.3f}ms "
                f"(overhead: {overhead:.3f}ms)"
            )
        return "\n".join(lines)


# ============================================================================
# Main Wrapper Class
# ============================================================================

class ValKeyBenchmarkWrapper:
    """Simple, elegant wrapper for valkey-search-benchmark.
    
    This class provides a clean interface for running benchmarks, parsing results,
    and implementing search algorithms (binary search, grid search) for finding
    optimal configurations.
    
    The wrapper leverages the C binary for:
    - Config persistence (.valkey-benchmark.conf)
    - Cluster mode detection and handling
    - Stage signaling for external monitoring tools
    
    Example:
        wrapper = ValKeyBenchmarkWrapper()
        config = BenchmarkConfig(host="localhost", dataset="data.bin")
        result = wrapper.run(config, operation="vec-query")
    """
    
    def __init__(self,
                 binary: Optional[str] = None,
                 verbose: bool = False,
                 display_output: bool = False):
        """Initialize the wrapper.

        Args:
            binary: Path to valkey-bench-rs binary (auto-detected if None)
            verbose: Enable verbose output
            display_output: Display benchmark output in real-time (default: False)

        Raises:
            BinaryNotFoundError: If binary cannot be found
        """
        self.binary = binary or self._find_binary()
        self.verbose = verbose
        self.display_output = display_output

        if self.verbose:
            print(f"Using binary: {self.binary}")
    
    def _find_binary(self) -> str:
        """Locate valkey-bench-rs binary.

        Search order:
        1. Environment variable: VALKEY_BENCHMARK_BIN
        2. Rust target directories (target/release, target/debug)
        3. System PATH

        Returns:
            Path to valkey-bench-rs binary

        Raises:
            BinaryNotFoundError: If binary not found
        """
        # Check environment variable
        if env_binary := os.getenv("VALKEY_BENCHMARK_BIN"):
            if Path(env_binary).exists():
                return env_binary

        # Check Rust target directories relative to project root
        project_root = Path(__file__).parent.parent.parent
        candidates = [
            project_root / "target/release/valkey-bench-rs",
            project_root / "target/debug/valkey-bench-rs",
        ]

        for path in candidates:
            if path.exists() and path.is_file():
                return str(path)

        # Try PATH
        result = subprocess.run(
            ["which", "valkey-bench-rs"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()

        raise BinaryNotFoundError(
            "valkey-bench-rs binary not found. "
            "Please build with 'cargo build --release' or set VALKEY_BENCHMARK_BIN environment variable."
        )
    
    def _detect_cluster(self, host: str) -> bool:
        """Detect if host is running in cluster mode.

        Uses valkey-bench-rs --cli to check cluster status.

        Args:
            host: Redis/Valkey server hostname

        Returns:
            True if cluster mode is enabled, False otherwise
        """
        try:
            result = subprocess.run(
                [self.binary, "--cli", "-h", host, "INFO", "Cluster"],
                capture_output=True,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            return "cluster_enabled:1" in result.stdout
        except (subprocess.TimeoutExpired, Exception):
            return False
    
    @contextmanager
    def stage(self, name: str, tag: Optional[str] = None):
        """Context manager for test stages (for external monitoring tools).
        
        Prints stage signals to stderr in a parseable format that external
        tools (like perf collectors) can monitor and react to.
        
        Note: The C binary (TODO #5) will also emit these signals. This wrapper
        emits them for wrapper-level stages (e.g., during binary search iterations).
        
        Args:
            name: Stage name (typically matches -t operation: vec-query, vec-load, etc.)
            tag: Optional tag for stage variant (e.g., "ef_100", "clients_20")
        
        Example:
            with wrapper.stage("vec-query", tag="ef_100"):
                result = wrapper.run(config)
        """
        stage_id = f"{name}:{tag}" if tag else name
        
        # Signal stage start to external monitoring tools
        print(f"[STAGE:START] {stage_id}", file=sys.stderr, flush=True)
        
        if self.verbose:
            print(f"\n=== Stage: {stage_id} ===")
        
        start = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start
            
            # Signal stage end
            print(
                f"[STAGE:END] {stage_id} duration={elapsed:.3f}s",
                file=sys.stderr,
                flush=True
            )
            
            if self.verbose:
                print(f"=== Completed {stage_id} in {elapsed:.2f}s ===\n")
    
    def run(self, 
            config: BenchmarkConfig,
            operation: Optional[str] = None) -> BenchmarkResult:
        """Run benchmark and return parsed results.
        
        Args:
            config: Benchmark configuration
            operation: Override config.operation if specified
        
        Returns:
            Parsed benchmark results
        
        Raises:
            BenchmarkError: If benchmark execution fails
            ParseError: If output parsing fails
        """
        # Use operation override if provided
        op = operation or config.operation
        
        # Build command: [binary] [prefix_args] [-t op] [to_args()]
        args = [self.binary] + config.prefix_args + ["-t", op] + config.to_args()
        
        # Auto-detect and add cluster flag if needed
        if self._detect_cluster(config.host):
            args.append("--cluster")
            if self.verbose:
                print("Detected cluster mode, adding --cluster flag")
        
        if self.verbose:
            print(f"Running: {' '.join(args)}")
        
        # Execute benchmark
        try:
            if self.display_output:
                # Stream output to terminal while capturing it
                process = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1  # Line buffered
                )
                
                output_lines = []
                stderr_lines = []
                
                # Read stdout line by line and display
                for line in process.stdout:
                    print(line, end='', flush=True)
                    output_lines.append(line)
                
                # Wait for process to complete and get stderr
                _, stderr = process.communicate()
                if stderr:
                    stderr_lines.append(stderr)
                
                # Check return code
                if process.returncode != 0:
                    raise BenchmarkError(
                        f"Benchmark failed with return code {process.returncode}\n"
                        f"stderr: {stderr}"
                    )
                
                # Reconstruct full output
                output = ''.join(output_lines)
            else:
                # Original behavior: capture all output silently
                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                if result.returncode != 0:
                    raise BenchmarkError(
                        f"Benchmark failed with return code {result.returncode}\n"
                        f"stderr: {result.stderr}"
                    )
                
                output = result.stdout
                
        except subprocess.TimeoutExpired:
            raise BenchmarkError("Benchmark timed out after 1 hour")
        except Exception as e:
            raise BenchmarkError(f"Failed to run benchmark: {e}")
        
        # Parse output
        return self._parse_output(output, config)
    
    def _parse_output(self, output: str, config: BenchmarkConfig) -> BenchmarkResult:
        """Parse benchmark console output.
        
        Parses the standard output format from valkey-benchmark to extract
        QPS, latency percentiles, and recall metrics.
        
        Args:
            output: Console output from valkey-benchmark
            config: Configuration used for the run
        
        Returns:
            Parsed benchmark results
        
        Raises:
            ParseError: If required metrics cannot be extracted
        """
        try:
            # Extract throughput (QPS)
            qps_match = re.search(r'throughput summary:\s+([\d.]+)\s+requests per second', output)
            if not qps_match:
                raise ParseError("Could not find throughput summary")
            qps = float(qps_match.group(1))
            
            # Extract latency metrics (in msec)
            # Look for the latency summary section
            latency_section = re.search(
                r'latency summary \(msec\):(.*?)(?:\n\n|\n  baseline|\Z)',
                output,
                re.DOTALL
            )
            if not latency_section:
                raise ParseError("Could not find latency summary")
            
            latency_text = latency_section.group(1)
            
            # Parse latency values: "  avg       min       p50       p95       p99       max\n"
            #                       "  1.234     0.567     1.111     2.345     3.456     5.678\n"
            latency_match = re.search(
                r'avg\s+min\s+p50\s+p95\s+p99\s+max\s*\n\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
                latency_text
            )
            if not latency_match:
                raise ParseError("Could not parse latency values")
            
            latency_avg = float(latency_match.group(1))
            latency_p50 = float(latency_match.group(3))
            latency_p95 = float(latency_match.group(4))
            latency_p99 = float(latency_match.group(5))
            latency_max = float(latency_match.group(6))
            
            # Extract baseline latency if present
            baseline_avg = None
            baseline_match = re.search(
                r'baseline network latency \(msec\):.*?avg.*?\n\s*([\d.]+)',
                output,
                re.DOTALL
            )
            if baseline_match:
                baseline_avg = float(baseline_match.group(1))
            
            # Extract recall metrics if present (for vec-query operations)
            recall_avg = None
            recall_min = None
            recall_max = None
            
            # Try optimizer output format first: "Recall Avg: 0.9847"
            recall_avg_match = re.search(r'Recall Avg:\s+([\d.]+)', output)
            if recall_avg_match:
                recall_avg = float(recall_avg_match.group(1))
            
            # Try standard recall statistics format
            recall_section = re.search(
                r'====== DATASET RECALL STATISTICS ======.*?Recall@\d+:(.*?)(?:\n\n|\Z)',
                output,
                re.DOTALL
            )
            if recall_section:
                recall_text = recall_section.group(1)
                
                # Parse: "    Average:  95.67% [Ext: 95.70%]"
                avg_match = re.search(r'Average:\s+([\d.]+)%', recall_text)
                if avg_match:
                    recall_avg = float(avg_match.group(1)) / 100.0
                
                # Parse: "    Min:      87.00% [Ext: 87.00%]"
                min_match = re.search(r'Min:\s+([\d.]+)%', recall_text)
                if min_match:
                    recall_min = float(min_match.group(1)) / 100.0
                
                # Parse: "    Max:      100.00% [Ext: 100.00%]"
                max_match = re.search(r'Max:\s+([\d.]+)%', recall_text)
                if max_match:
                    recall_max = float(max_match.group(1)) / 100.0
            
            return BenchmarkResult(
                qps=qps,
                latency_avg=latency_avg,
                latency_p50=latency_p50,
                latency_p95=latency_p95,
                latency_p99=latency_p99,
                latency_max=latency_max,
                recall_avg=recall_avg,
                recall_min=recall_min,
                recall_max=recall_max,
                baseline_latency_avg=baseline_avg,
                config=config,
            )
            
        except (AttributeError, ValueError, IndexError) as e:
            raise ParseError(f"Failed to parse benchmark output: {e}\nOutput:\n{output}")
    
    def save_results_csv(self, results: List[BenchmarkResult], output_file: str):
        """Save results to CSV file.
        
        Args:
            results: List of benchmark results
            output_file: Path to output CSV file
        """
        if not results:
            return
        
        with open(output_file, 'w', newline='') as f:
            writer = csv_module.writer(f)
            
            # Write header
            header = [
                'timestamp', 'qps', 'latency_avg', 'latency_p50', 'latency_p95',
                'latency_p99', 'latency_max', 'recall_avg', 'recall_min', 'recall_max',
                'baseline_latency', 'num_clients', 'num_threads', 'ef_search'
            ]
            writer.writerow(header)
            
            # Write data
            for result in results:
                row = [
                    result.timestamp,
                    result.qps,
                    result.latency_avg,
                    result.latency_p50,
                    result.latency_p95,
                    result.latency_p99,
                    result.latency_max,
                    result.recall_avg if result.recall_avg is not None else '',
                    result.recall_min if result.recall_min is not None else '',
                    result.recall_max if result.recall_max is not None else '',
                    result.baseline_latency_avg if result.baseline_latency_avg is not None else '',
                    result.config.num_clients if result.config else '',
                    result.config.num_threads if result.config else '',
                    result.config.ef_search if result.config and result.config.ef_search else '',
                ]
                writer.writerow(row)
        
        if self.verbose:
            print(f"Results saved to {output_file}")
    
    # ========================================================================
    # Search Algorithms
    # ========================================================================
    
    def binary_search_max_qps(self,
                               base_config: BenchmarkConfig,
                               target_recall: float,
                               param_name: str = "num_clients",
                               min_val: int = 1,
                               max_val: int = 200,
                               tolerance: int = 2) -> Optional[BenchmarkResult]:
        """Binary search to find maximum QPS at target recall threshold.
        
        Searches for the optimal parameter value (typically num_clients or ef_search)
        that maximizes QPS while maintaining the target recall threshold.
        
        Args:
            base_config: Base configuration to modify
            target_recall: Minimum required recall (0.0-1.0)
            param_name: Parameter to sweep ("num_clients", "ef_search", "num_threads")
            min_val: Minimum parameter value
            max_val: Maximum parameter value
            tolerance: Stop when search range <= tolerance
        
        Returns:
            Best result achieving target recall, or None if target unreachable
        
        Example:
            config = BenchmarkConfig(host="localhost", dataset="data.bin")
            best = wrapper.binary_search_max_qps(
                config, 
                target_recall=0.95,
                param_name="num_clients",
                max_val=100
            )
        """
        best_result = None
        left, right = min_val, max_val
        
        if self.verbose:
            print(f"\nBinary search: {param_name} in [{left}, {right}] "
                  f"for recall >= {target_recall:.2%}")
        
        while right - left > tolerance:
            mid = (left + right) // 2
            
            # Create test configuration
            if param_name == "num_clients":
                test_config = replace(base_config, num_clients=mid)
            elif param_name == "ef_search":
                test_config = replace(base_config, ef_search=mid)
            elif param_name == "num_threads":
                test_config = replace(base_config, num_threads=mid)
            else:
                raise ValueError(f"Unknown parameter: {param_name}")
            
            # Run benchmark with stage signaling
            with self.stage(base_config.operation, tag=f"{param_name}_{mid}"):
                try:
                    result = self.run(test_config)
                except BenchmarkError as e:
                    if self.verbose:
                        print(f"  {param_name}={mid}: Failed - {e}")
                    right = mid - 1
                    continue
            
            # Check if recall meets threshold
            if result.recall_avg is None:
                raise BenchmarkError(
                    "Operation does not produce recall metrics. "
                    "Use vec-query or similar operation."
                )
            
            meets_target = result.recall_avg >= target_recall
            
            if self.verbose:
                status = "✓" if meets_target else "✗"
                print(f"  {status} {param_name}={mid}: "
                      f"QPS={result.qps:.1f}, Recall={result.recall_avg:.2%}")
            
            if meets_target:
                # Recall is good, try higher parameter value for more QPS
                if best_result is None or result.qps > best_result.qps:
                    best_result = result
                left = mid + 1
            else:
                # Recall too low, need higher parameter value
                right = mid - 1
        
        if best_result and self.verbose:
            print(f"\n✓ Best: {param_name}={getattr(best_result.config, param_name)}, "
                  f"QPS={best_result.qps:.1f}, Recall={best_result.recall_avg:.2%}")
        elif self.verbose:
            print(f"\n✗ Could not achieve target recall {target_recall:.2%}")
        
        return best_result
    
    def grid_search(self,
                    base_config: BenchmarkConfig,
                    param_grid: Dict[str, List[Any]],
                    filter_fn: Optional[Callable[[BenchmarkResult], bool]] = None) -> List[BenchmarkResult]:
        """Grid search over parameter combinations.
        
        Exhaustively tests all combinations of parameters in the grid.
        Optionally filters results based on custom criteria.
        
        Args:
            base_config: Base configuration to modify
            param_grid: Dictionary mapping parameter names to lists of values
                       e.g., {"num_clients": [10, 20, 50], "ef_search": [50, 100]}
            filter_fn: Optional function to filter results (return True to keep)
        
        Returns:
            List of all results (or filtered results)
        
        Example:
            config = BenchmarkConfig(host="localhost", dataset="data.bin")
            results = wrapper.grid_search(
                config,
                param_grid={
                    "num_clients": [10, 20, 50, 100],
                    "ef_search": [50, 100, 200]
                },
                filter_fn=lambda r: r.recall_avg >= 0.95  # Only high recall
            )
        """
        import itertools
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        combinations = list(itertools.product(*param_values))
        
        if self.verbose:
            print(f"\nGrid search: {len(combinations)} combinations")
            for name in param_names:
                print(f"  {name}: {param_grid[name]}")
        
        results = []
        
        for i, values in enumerate(combinations, 1):
            # Create configuration for this combination
            params = dict(zip(param_names, values))
            
            # Build config with params
            test_config = base_config
            for param_name, param_value in params.items():
                if param_name == "num_clients":
                    test_config = replace(test_config, num_clients=param_value)
                elif param_name == "ef_search":
                    test_config = replace(test_config, ef_search=param_value)
                elif param_name == "num_threads":
                    test_config = replace(test_config, num_threads=param_value)
                elif param_name == "num_requests":
                    test_config = replace(test_config, num_requests=param_value)
                else:
                    # For other params, add to extra_args
                    extra = test_config.extra_args.copy()
                    extra.extend([f"--{param_name.replace('_', '-')}", str(param_value)])
                    test_config = replace(test_config, extra_args=extra)
            
            # Create tag for this combination
            tag = "_".join(f"{k}_{v}" for k, v in params.items())
            
            # Run benchmark
            with self.stage(test_config.operation, tag=tag):
                try:
                    result = self.run(test_config)
                    results.append(result)
                    
                    if self.verbose:
                        recall_str = f", Recall={result.recall_avg:.2%}" if result.recall_avg else ""
                        print(f"  [{i}/{len(combinations)}] {tag}: "
                              f"QPS={result.qps:.1f}, Latency={result.latency_p99:.3f}ms{recall_str}")
                
                except BenchmarkError as e:
                    if self.verbose:
                        print(f"  [{i}/{len(combinations)}] {tag}: Failed - {e}")
        
        # Apply filter if provided
        if filter_fn:
            filtered = [r for r in results if filter_fn(r)]
            if self.verbose:
                print(f"\nFiltered: {len(filtered)}/{len(results)} results")
            return filtered
        
        return results
    
    def find_max_qps_with_constraints(self,
                                      base_config: BenchmarkConfig,
                                      min_recall: float,
                                      max_latency_p99: Optional[float] = None,
                                      param_name: str = "num_clients",
                                      min_val: int = 1,
                                      max_val: int = 200) -> Optional[BenchmarkResult]:
        """Find maximum QPS with recall and latency constraints.
        
        Similar to binary_search_max_qps but adds optional latency constraint.
        
        Args:
            base_config: Base configuration
            min_recall: Minimum required recall (0.0-1.0)
            max_latency_p99: Maximum allowed P99 latency in ms (optional)
            param_name: Parameter to sweep
            min_val: Minimum parameter value
            max_val: Maximum parameter value
        
        Returns:
            Best result meeting all constraints, or None
        
        Example:
            # Find max QPS with recall >= 95% and P99 <= 10ms
            best = wrapper.find_max_qps_with_constraints(
                config,
                min_recall=0.95,
                max_latency_p99=10.0,
                param_name="num_clients"
            )
        """
        def meets_constraints(result: BenchmarkResult) -> bool:
            """Check if result meets all constraints."""
            if result.recall_avg is None or result.recall_avg < min_recall:
                return False
            if max_latency_p99 and result.latency_p99 > max_latency_p99:
                return False
            return True
        
        best_result = None
        left, right = min_val, max_val
        
        if self.verbose:
            constraints = [f"recall >= {min_recall:.2%}"]
            if max_latency_p99:
                constraints.append(f"p99 <= {max_latency_p99:.3f}ms")
            print(f"\nSearching {param_name} in [{left}, {right}] "
                  f"with constraints: {', '.join(constraints)}")
        
        while right - left > 2:
            mid = (left + right) // 2
            
            # Create test configuration
            if param_name == "num_clients":
                test_config = replace(base_config, num_clients=mid)
            elif param_name == "ef_search":
                test_config = replace(base_config, ef_search=mid)
            elif param_name == "num_threads":
                test_config = replace(base_config, num_threads=mid)
            else:
                raise ValueError(f"Unknown parameter: {param_name}")
            
            # Run benchmark
            with self.stage(base_config.operation, tag=f"{param_name}_{mid}"):
                try:
                    result = self.run(test_config)
                except BenchmarkError as e:
                    if self.verbose:
                        print(f"  {param_name}={mid}: Failed - {e}")
                    right = mid - 1
                    continue
            
            # Check constraints
            if meets_constraints(result):
                if best_result is None or result.qps > best_result.qps:
                    best_result = result
                
                if self.verbose:
                    print(f"  ✓ {param_name}={mid}: QPS={result.qps:.1f}, "
                          f"Recall={result.recall_avg:.2%}, P99={result.latency_p99:.3f}ms")
                
                left = mid + 1  # Try higher for more QPS
            else:
                recall_str = f"Recall={result.recall_avg:.2%}" if result.recall_avg else "No recall"
                if self.verbose:
                    print(f"  ✗ {param_name}={mid}: {recall_str}, "
                          f"P99={result.latency_p99:.3f}ms")
                right = mid - 1
        
        if best_result and self.verbose:
            print(f"\n✓ Best: {param_name}={getattr(best_result.config, param_name)}, "
                  f"QPS={best_result.qps:.1f}, Recall={best_result.recall_avg:.2%}, "
                  f"P99={best_result.latency_p99:.3f}ms")
        
        return best_result
