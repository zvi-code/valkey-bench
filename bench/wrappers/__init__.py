"""
Simple, elegant wrapper framework for valkey-search-benchmark.

This package provides a clean Python interface for building sophisticated
benchmark wrappers while keeping the implementation simple and maintainable.

Example usage:
    from wrappers import ValKeyBenchmarkWrapper, BenchmarkConfig
    
    wrapper = ValKeyBenchmarkWrapper(verbose=True)
    config = BenchmarkConfig(
        host="localhost",
        dataset="datasets/sift-128.bin",
        num_clients=20,
        ef_search=100
    )
    
    with wrapper.stage("vec-query", tag="ef_100"):
        result = wrapper.run(config, operation="vec-query")
    
    print(f"QPS: {result.qps}, Recall: {result.recall_avg:.2%}")
"""

from .base_wrapper import (
    ValKeyBenchmarkWrapper,
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkError,
    BinaryNotFoundError,
    ParseError,
    find_dataset_path,
    detect_dataset_info,
    generate_index_name,
    generate_search_prefix,
)

__all__ = [
    'ValKeyBenchmarkWrapper',
    'BenchmarkConfig',
    'BenchmarkResult',
    'BenchmarkError',
    'BinaryNotFoundError',
    'ParseError',
    'find_dataset_path',
    'detect_dataset_info',
    'generate_index_name',
    'generate_search_prefix',
]

__version__ = '0.1.0'
