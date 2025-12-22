# Runtime Configuration Management

The valkey-benchmark tool supports applying server-side configurations before running benchmarks, allowing you to test performance under different server settings without manually changing configurations.

## Overview

Runtime configuration management allows you to:
- Apply server configurations from a simple config file before benchmarks
- Test performance with different server settings automatically
- Optionally restore original configurations after benchmarks complete
- Apply configurations to all nodes in a cluster simultaneously

## Usage

### Basic Usage

```bash
# Apply runtime config before benchmark
./valkey-benchmark -h localhost -t ping -n 10000 \
    --runtime-config my-config.conf

# Apply and restore original config after benchmark
./valkey-benchmark -h localhost -t ping -n 10000 \
    --runtime-config my-config.conf \
    --restore-config
```

### With Search Benchmarks

```bash
# Test vector search with optimized server settings
./valkey-benchmark -h localhost --cluster \
    --dataset openai-large-5m.bin \
    -t vec-query --search --vector-dim 1536 \
    --search-name openai_5m --search-prefix zvec_: \
    -n 10000 -c 20 --threads 10 \
    --runtime-config search-optimized.conf \
    --restore-config
```

## Configuration File Format

The configuration file uses a simple key-value format:

```conf
# Comments start with #
# Empty lines are ignored
# Format: key value (or key=value)

# IO Threads
io-threads 8
io-threads-do-reads yes

# Memory Settings
maxmemory 10gb
maxmemory-policy allkeys-lru

# Network Settings
tcp-backlog 2048
tcp-keepalive 300
timeout 0

# Persistence (disable for benchmarks)
save ""
appendonly no

# Slow Log
slowlog-log-slower-than 10000
slowlog-max-len 128

# Client Limits
maxclients 50000
```

## Common Configuration Examples

### High-Throughput Configuration

```conf
# high-throughput.conf
io-threads 8
io-threads-do-reads yes
tcp-backlog 4096
maxclients 50000
save ""
appendonly no
```

### Low-Latency Configuration

```conf
# low-latency.conf
io-threads 4
io-threads-do-reads yes
tcp-keepalive 60
timeout 0
```

### Vector Search Optimized

```conf
# vector-search.conf
io-threads 8
io-threads-do-reads yes
maxmemory 20gb
maxmemory-policy noeviction
save ""
appendonly no
tcp-backlog 4096
maxclients 50000
```

## Behavior

### Configuration Application

- **Cluster Mode**: Configurations are applied to ALL cluster nodes
- **Standalone Mode**: Configurations are applied to the single node
- **Original Values**: Original values are saved when configurations are applied
- **Verbose Output**: Shows each configuration change (unless `--quiet` is used)

### Output Example

```
=== Applying Runtime Configuration ===
Configuration entries: 5
✓ Set io-threads = 8 on node1:6379 (was: 1)
✓ Set timeout = 300 on node1:6379 (was: 0)
✓ Set tcp-keepalive = 60 on node1:6379 (was: 300)
✓ Set slowlog-log-slower-than = 5000 on node1:6379 (was: 10000)
✓ Set slowlog-max-len = 64 on node1:6379 (was: 128)
Total configurations applied: 5
=====================================

Successfully applied 5 runtime configuration settings
```

### Restoration

When `--restore-config` is used:

1. Original configuration values are saved during application
2. After benchmarks complete, original values are restored
3. Restoration happens automatically even if benchmark fails

```
=== Restoring Original Configuration ===
✓ Restored io-threads = 1 on node1:6379
✓ Restored timeout = 0 on node1:6379
✓ Restored tcp-keepalive = 300 on node1:6379
✓ Restored slowlog-log-slower-than = 10000 on node1:6379
✓ Restored slowlog-max-len = 128 on node1:6379
Total configurations restored: 5
========================================

Successfully restored 5 configuration settings
```

## Error Handling

- **Invalid Config Key**: Warning printed, benchmark continues with other configs
- **Missing File**: Warning printed, benchmark runs without runtime config
- **Permission Errors**: Error printed, benchmark continues
- **Connection Errors**: Error printed per node

## Best Practices

1. **Test Configurations First**: Test configurations on a non-production system
2. **Use Restoration**: Always use `--restore-config` for safety
3. **Disable Persistence**: For benchmarks, disable `save` and `appendonly`
4. **Monitor Memory**: Be careful with `maxmemory` settings
5. **Document Changes**: Keep comments in config files explaining settings

## Supported Configuration Parameters

Any `CONFIG SET` compatible parameter can be used. Common parameters include:

**Network**:
- `io-threads`, `io-threads-do-reads`
- `tcp-backlog`, `tcp-keepalive`
- `timeout`, `maxclients`

**Memory**:
- `maxmemory`, `maxmemory-policy`
- `maxmemory-samples`

**Persistence**:
- `save`, `appendonly`
- `appendfsync`, `no-appendfsync-on-rewrite`

**Logging**:
- `loglevel`, `slowlog-log-slower-than`
- `slowlog-max-len`

**Performance**:
- `lazyfree-lazy-eviction`
- `lazyfree-lazy-expire`
- `lazyfree-lazy-server-del`

Refer to Valkey/Redis documentation for full list of CONFIG SET parameters.

## Integration with Wrappers

Runtime configuration can be combined with wrapper scripts for automated testing:

```bash
# test-configurations.sh
#!/bin/bash

for config in configs/*.conf; do
    echo "Testing with $(basename $config)"
    ./valkey-benchmark -h $HOST \
        -t vec-query \
        --dataset openai-large-5m.bin \
        --runtime-config "$config" \
        --restore-config \
        -n 10000 -c 20
done
```

## Troubleshooting

**Problem**: Configuration not applied  
**Solution**: Check if parameter name is correct using `CONFIG GET <param>`

**Problem**: Permission denied errors  
**Solution**: Ensure the user has CONFIG SET permissions on the server

**Problem**: Configuration persists after benchmark  
**Solution**: Use `--restore-config` flag

**Problem**: Different values on different nodes  
**Solution**: This is expected for cluster-specific settings; check per-node configs

## See Also

- [BENCHMARKING.md](BENCHMARKING.md) - Benchmarking workflows
- [ADVANCED.md](ADVANCED.md) - Advanced features
- [TODO.md](TODO.md) - Planned enhancements
