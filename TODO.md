# TODO: Enhancements and Planned Features

This document tracks planned enhancements and feature ideas for the valkey-search-benchmark project.

## Rust rewrite TODO's

### 1. Make sure optimizer works for all kinds of target goals
**Status:** Planned
**Description:** support client+thread optimal point for latency, for qps, for latency&qps. For search workloads
**Benefits:** 

### 2. Tags support
**Status:** Planned
**Description:** Import functionality from c code
**Benefits:** 

### 2. Tags support
**Status:** Planned
**Description:** Import functionality from c code
**Benefits:** 

## Loader Enhancements

### 1. Ground Truth Query Vector Insertion
**Status:** Planned
**Description:** Add option to insert all ground truth query vectors into the index.  
**Benefits:** Enables more comprehensive testing and validation scenarios.

### 2. Index Configuration Verification
**Status:** Planned
**Description:** Verify index configuration for existing indexes. If the configuration doesn't match what's requested, provide option to clean/recreate the index.  
**Benefits:** Prevents test failures due to configuration mismatches and ensures consistency.

### 3. Ground Truth Generation via Flat Search
**Status:** Planned
**Description:** Generate ground truth vectors using flat search for existing indexes.  
**Benefits:** Allows creation of ground truth data without external dependencies.

### 4. ~~Runtime Configuration Management~~
**Status:** ✅ Completed  
**Description:** Add ability to set server-side configurations before a test is run and restore original configurations afterward.
**Benefits:** More flexible testing scenarios without manual server configuration changes. Enables automated testing of different server configurations.

### 5. Test Stage and Tag Reporting
**Status:** Planned
**Description:** Report `test stage` and `test tag` for external tools to collect profiling data.  
**Benefits:** Better integration with profiling and monitoring tools; easier correlation of metrics with test phases.

### 6. ~~Persistent Configuration Storage~~
**Status:** Done  
**Description:** Save last configuration in a file. If file exists, use the configuration for any argument not provided by the user.  
**Exclusions:** Should NOT include:
- `-t` option (the command)
- `-h` argument (the host)

**Benefits:** Reduces repetitive command-line arguments and improves user experience.

### 7. Expiry Support
**Status:** Planned
**Description:** Add ability to set TTL/expiry on inserted vectors.  
**Benefits:** Test scenarios involving data expiration and cache eviction.

### 8. Numeric and Tag Filters
**Status:** In Progress
**Description:** Add ability to set numeric and tag filters for queries.  
**Benefits:** Test filtered search scenarios and mixed workloads.

### 9. Variable Tag Lengths
**Status:** In Progress
**Description:** Add ability to set different tag lengths for testing.  
**Benefits:** Test impact of metadata size on performance.

### 10. Dynamic Thread and Client Scaling
**Status:** Planned  
**Description:** Add dynamic thread and client count adjustment without terminating threads.  
**Benefits:** Test dynamic scaling scenarios and eliminate warmup overhead between tests.

### 11. Valkey Logic Encapsulation
**Status:** Started
**Description:** Encapsulate Valkey-specific logic to prepare for future extensions to other vector databases.  
**Benefits:** Better code organization and easier support for multiple backends.

### 17. Dataset Extension
**Status:** Started
**Description:** Extend dataset by shifting existing dataset by the diameter of current dataset.  
**Note:** Will not work for cosine similarity.  
**Benefits:** Create larger synthetic datasets from existing ones.

### 18. JSON Support
**Status:** Planned  
**Description:** Add support for search data in JSON data type (HASH is the default).
**Benefits:** Improved usability and integration with external tools.

## 19. Allow Range based address specification
**Status:** Started
**Description:** Allow specifying a range of addresses not starting from 0.  
**Benefits:** More flexible data addressing schemes.

### 20. Support Multiple Indexes
**Status:** Planned  
**Description:** Add support for multiple indexes in a single benchmark run.  
**Benefits:** Test scenarios involving multiple indexes and their interactions.

### 21. Support nested key addressing
**Status:** Planned  
**Description:** Support nested key addressing like fields in hash or json data types. Allow specifying field ranges per key. So you can run random hset load, with varying number of fields in key and length.
**Benefits:** More realistic testing scenarios with complex data structures.

### 22. Extend support of delete operations
**Status:** Started
**Description:** Extend delete operations to support deleting by range, by % or by capacity target.  
**Benefits:** More flexible data management scenarios.

### 23. Support for mixed workloads + hybrid search
**Status:** Started
**Description:** Add support for mixed workloads involving different operation types (e.g., search, insert, delete) in a single benchmark run. This will also support having non-search operations running in the background while search operations are being benchmarked.
**Benefits:** More realistic testing scenarios that mimic production workloads.

### 24. ~~Evaluate base latency~~
**Status:** ✅ Completed  
**Description:** Measure baseline network latency using PING commands at the beginning of a benchmark run. The baseline is measured with 10,000 PING operations using single-threaded, single-client configuration to establish pure network RTT. The baseline latency is displayed separately and included in both console output and CSV exports, showing processing overhead (operation latency - baseline latency).
**Benefits:** More accurate latency measurements by separating network overhead from operation-specific processing time.
**Implementation:** Enabled by default (no flags needed). Runs silently before benchmarks. Use `--no-baseline` to disable. Results include avg, p50, p90, p95, p99, and max latencies in both console and CSV output.

### 25. Support multiple clusters
**Status:** Planned  
**Description:** Add support for benchmarking across multiple Valkey clusters simultaneously.  
**Benefits:** Enables testing of distributed scenarios and cluster interactions.

### 26. Collect latency per-node both in CME and when using replicas in CMD
**Status:** Started
**Description:** Extend the benchmark tool to collect and report latency metrics for each individual node in a cluster environment, both in Cluster Mode Enabled (CME) and when using replicas in Cluster Mode Disabled (CMD).  
**Benefits:** Provides deeper insights into performance variations across different nodes, helping identify bottlenecks and optimize cluster configurations.

### 27. Add better handling for out of sync replicas
**Status:** Planned  
**Description:** Improve the benchmark tool's handling of out-of-sync replicas by implementing detection mechanisms and appropriate response strategies. This may include skipping lagging replicas during benchmarks or providing detailed reporting on their status.  
**Benefits:** Ensures more accurate benchmark results by avoiding the influence of lagging replicas and providing clearer insights into cluster health.

### 28. Implement vec-scan-q-verify
**Status:** Planned
**Description:** Implement a new operation type `vec-scan-q-verify` that performs vector scan queries and verifies the results against self. This can be used for datasets without ground truth. We can generate "statistical" recall calculations when looking on aggregate queries. How many top-k (k=1) we got out on N queries.
**Benefits:** Some sanity test of quality, without the cost of ground truth generation.

### 29. Embed ef-construction and m setting in the index name
**Status:** Planned
**Description:** Modify the index naming convention to include `ef-construction` and `m` parameters directly in the index name. This will help in easily identifying the configuration of each index based on its name.  
**Benefits:** Simplifies index management and identification, especially when dealing with multiple indexes with different configurations

### 30. Deduce vector-field-name from the dataset name. 
**Status:** Planned
**Description:** Implement a mechanism to automatically deduce the vector field name from the dataset name during index creation. This will streamline the process of setting up indexes by reducing the need for manual configuration.
**Benefits:** Enhances usability and reduces configuration errors by automating the mapping between datasets and their corresponding vector fields.

### 31. Vector load progress reporting
**Status:** Planned
**Description:** Add progress reporting for vector load operations, providing real-time feedback on the status of data loading. This could include percentage completion, estimated time remaining, and current throughput.
**Benefits:** Improves user experience by keeping users informed about long-running load operations and helps in monitoring performance.

### 32. Improve node workload distribution in CME\CMD
**Status:** Planned
**Description:** Enhance the workload distribution mechanism in both Cluster Mode Enabled (CME) and Cluster Mode Disabled (CMD) to ensure more balanced resource utilization across nodes.
**Benefits:** Optimizes performance and resource usage, reducing the risk of bottlenecks and improving overall system efficiency.

### 33. When Ctrl+c is pressed when running load, generate output for the so far execution

------
## Search Quality Metrics Enhancements
### Add additional search results quality metrics
**Status:** Planned
**Description:** Implement additional metrics to evaluate the quality of search results beyond simple recall. These metrics provide deeper insights into ranking quality and relevance ordering:

#### 🧮 1. **Mean Average Precision (MAP)**

**Definition:**
For each query, compute the *average precision* (AP), which takes into account the *rank positions* of the correct items. Then take the mean across queries.

```
AP = (1 / N_relevant) * Σ(k=1 to K) P@k · rel(k)
```

where `P@k` is the precision at rank `k`, and `rel(k)` is 1 if the item at rank `k` is relevant.

MAP rewards algorithms that return correct items early in the ranking list — not just within top-k but near the top.

**Use case:** Common in IR (information retrieval) and vector search evaluation when ranking quality matters.

---

#### 📈 2. **Normalized Discounted Cumulative Gain (NDCG)**

**Definition:**
Considers not just binary relevance (hit/miss) but also *graded* relevance (e.g., true rank distance).

```
DCG@k = Σ(i=1 to k) (2^rel_i - 1) / log₂(i + 1)
NDCG@k = DCG@k / IDCG@k
```

Here, a relevant item appearing at rank 2 contributes less than at rank 1 due to the logarithmic discount. If your true neighbors are far down (e.g., rank 100,000), their contribution will be near zero.

**Use case:** Standard metric in ranking and recommendation systems — measures how *well ordered* your retrieved results are.

---

#### 📊 3. **Mean Reciprocal Rank (MRR)**

**Definition:**
Measures how soon the *first* relevant item appears in the ranked list:

```
MRR = (1 / N) * Σ(i=1 to N) (1 / rank_i)
```

Useful when you mostly care about whether at least one good match is near the top.

---

#### 💡 4. **Rank-weighted Recall or Recall@R**

Some papers also compute recall not just for the top-k results, but for *different depth thresholds* (e.g., recall@10, recall@100, recall@1000).
Plotting this as a curve helps visualize how far the true neighbors are distributed.

**Benefits:** 
- More comprehensive evaluation of search quality
- Ranking-aware metrics (MAP, NDCG, MRR) complement recall
- Better understanding of result quality at different depth thresholds
- Industry-standard metrics for comparison with other systems

**Implementation:** Add these metrics alongside existing recall calculations, with options to enable/disable specific metrics and export results to CSV for analysis.

---
## Wrapper Scripts Enhancements

### 0. ~~Implement infrastructure for wrappers as described in WRAPPERS.md~~
**Status:** ✅ Completed  
**Description:** Established the foundational infrastructure to support the development and integration of various wrapper scripts as outlined in WRAPPERS.md.  

**Implementation:** Created Python-based wrapper framework following KIS (Keep It Simple) principles:
- **bench/wrappers/base_wrapper.py**: Single-file core framework (~580 lines) with `ValKeyBenchmarkWrapper`, `BenchmarkConfig`, `BenchmarkResult` classes
- **Search algorithms**: `binary_search_max_qps()`, `grid_search()`, `find_max_qps_with_constraints()`
- **Stage signaling**: Simple context manager emits `[STAGE:START/END]` signals to stderr for external monitoring tools
- **Auto-detection**: Binary location, cluster mode detection
- **Result parsing**: Console output and CSV export
- **Example scripts**: `max_qps_recall.py` (Wrapper #1 implementation), `stage-monitor.sh` (external monitoring demo)

**Design Decisions:**
- Leverage C binary's config persistence, cluster handling, and stage signaling (TODO #5)
- No separate ConfigManager, ClusterManager, or StageManager modules - kept in base class
- External tools parse stage signals rather than embedded perf collection
- All core logic in single file for maintainability

**Benefits:** Clean, maintainable foundation for all future wrapper implementations. Ready for Wrappers #2-6.

**Documentation:** See `bench/wrappers/README.md` for full API reference and usage examples.

### 1. ~~Max QPS at Target Recall~~
**Status:** ✅ Completed (via infrastructure)  
**Description:** Add wrapper to find maximum QPS achievable at a specified recall threshold.  
**Implementation:** `bench/scripts/max_qps_recall.py` - Full CLI tool with binary search algorithm.  
**Benefits:** Automated performance envelope discovery.

### 2. ~~Max QPS at Recall and Latency Thresholds~~
**Status:** ✅ Completed (via infrastructure)  
**Description:** Add wrapper to find maximum QPS while maintaining both recall and latency thresholds.  
**Implementation:** `ValKeyBenchmarkWrapper.find_max_qps_with_constraints()` method and available in `max_qps_recall.py` via `--max-p99-latency` flag.  
**Benefits:** More realistic performance testing with SLA constraints.

### 3. ~~Optimal Configuration Discovery~~
**Status:** ✅ Completed (via infrastructure)  
**Description:** Add wrapper to find optimal thread and client count configurations.  
**Implementation:** `ValKeyBenchmarkWrapper.grid_search()` method with filtering support.  
**Benefits:** Automated tuning for specific hardware and workload combinations.

### 4. ~~Profiling Integration with Test Stages~~
**Status:** ✅ Completed (via infrastructure)  
**Description:** Use `test stage` and `test tag` to trigger data collectors while in specific stages and tag the output accordingly.  
**Implementation:** `bench/scripts/stage-monitor.sh` - External bash script that monitors `[STAGE:START/END]` signals and triggers perf collection during specified stages.  
**Benefits:** Automated profiling workflow with properly labeled data.

### 5. Memory Saturation Testing
**Status:** Planned
**Description:** Wrapper for testing with 100% memory utilization.  
**Benefits:** Test behavior under memory pressure and eviction scenarios.

### 6. Payload Impact Testing
**Status:** Planned
**Description:** Add wrapper to test the impact of different payload sizes and types.  
**Benefits:** Understand memory and performance tradeoffs with different metadata configurations.

---

## Priority Levels (TBD)

Items should be prioritized based on:
- User demand
- Implementation complexity
- Dependencies between features
- Impact on testing capabilities

## Contributing

When implementing any of these features:
1. Update this document with implementation status
2. Add relevant documentation to README.md or ADVANCED.md
3. Include tests where applicable
4. Update BENCHMARKING.md if the feature affects benchmarking workflows

---

## Commands examples
```
/home/ubuntu/valkey-search-benchmark/build-debug/bin/valkey-benchmark -h $HOST --rfr all  --dataset /home/ubuntu/valkey-search-benchmark/datasets/openai-large-5m.bin -t vec-query --search --vector-dim 1536 --search-name openai-large-5m-5M-1536-100 --search-prefix zvec_openai5m: -n 60000 -c 800 --ef-search 256 --nocontent --no-save-config --k 100  --balance-nodes --balance-quota-step 10000 --threads 10 --runtime-config "./16xl-ec2.conf"
```

```
/home/ubuntu/valkey-search-benchmark/build-debug/bin/valkey-benchmark -h $HOST --rfr no --dataset /home/ubuntu/valkey-search-benchmark/datasets/openai-large-5m.bin -t vec-load --search --vector-dim 1536 --search-name openai-large-5m-5M-1536-100 --search-prefix zvec_openai5m: -n 5000000 -c 440 --m 16 --ef-construction 256 --nocontent
```

*Last Updated: October 21, 2025*
