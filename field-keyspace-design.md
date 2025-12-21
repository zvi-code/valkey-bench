# Extended Field and Keyspace Configuration Design

## Overview

This document outlines the design for extending the benchmark tool to support:
1. Multiple tag and numeric fields with distributions
2. Numeric value ranges with type-aware formatting
3. Generic keyspace operations with field-level access

---

## 1. Numeric Field Enhancements

### Current State
- Single `numeric_field: Option<String>` in SearchConfig
- Fixed 12-digit format, always uses key_num as value
- No distribution control

### Proposed Design

```rust
/// Numeric field configuration
#[derive(Debug, Clone)]
pub struct NumericFieldConfig {
    /// Field name (user-defined)
    pub name: String,
    /// Value type (affects formatting and parsing)
    pub value_type: NumericValueType,
    /// Distribution for value generation
    pub distribution: NumericDistribution,
    /// Minimum value (inclusive)
    pub min: f64,
    /// Maximum value (exclusive)
    pub max: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum NumericValueType {
    /// Integer (formatted as decimal)
    Int,
    /// Floating point (formatted with precision)
    Float { precision: u8 },
    /// Unix timestamp (seconds since epoch)
    UnixTimestamp,
    /// ISO 8601 date-time string
    IsoDateTime,
    /// Date only (YYYY-MM-DD)
    DateOnly,
}

#[derive(Debug, Clone)]
pub enum NumericDistribution {
    /// Uniform random distribution
    Uniform,
    /// Zipfian distribution (power-law, models "hot" values)
    Zipfian { skew: f64 },
    /// Normal/Gaussian distribution
    Normal { mean: f64, stddev: f64 },
    /// Constant value (useful for testing)
    Constant { value: f64 },
    /// Sequential (increments with each key)
    Sequential { start: f64, step: f64 },
    /// Key-based (deterministic based on key_num)
    KeyBased,
}
```

### CLI Syntax

```bash
# Single numeric field (backwards compatible)
--numeric-field timestamp

# Extended syntax with type and distribution
--numeric-field "timestamp:unix:uniform:1609459200:1704067200"
--numeric-field "price:float2:normal:100.0:15.0"  # mean=100, stddev=15
--numeric-field "score:int:zipfian:0.99:0:1000"   # skew=0.99, range 0-1000

# Multiple numeric fields
--numeric-fields "timestamp:unix:uniform:1609459200:1704067200,score:int:uniform:0:100"
```

### Parsing Format

```
field_name:type:distribution[:params...]

Types:
  int          - Integer
  floatN       - Float with N decimal places (e.g., float2)
  unix         - Unix timestamp
  iso          - ISO 8601 datetime
  date         - Date only (YYYY-MM-DD)

Distributions:
  uniform:min:max
  zipfian:skew:min:max
  normal:mean:stddev
  const:value
  seq:start:step
  key           - Uses key_num as seed
```

---

## 2. Multiple Tag and Numeric Fields

### Current State
- Single `tag_field: Option<String>` with `tag_distributions: Option<TagDistributionSet>`
- Single `numeric_field: Option<String>`

### Proposed Design

```rust
/// Collection of field configurations for vector search
#[derive(Debug, Clone, Default)]
pub struct FieldConfigs {
    /// Tag fields with their distributions
    pub tag_fields: Vec<TagFieldConfig>,
    /// Numeric fields with their distributions
    pub numeric_fields: Vec<NumericFieldConfig>,
}

/// Tag field configuration
#[derive(Debug, Clone)]
pub struct TagFieldConfig {
    /// Field name
    pub name: String,
    /// Tag value distributions
    pub distributions: TagDistributionSet,
}

impl FieldConfigs {
    /// Total byte length needed for all field placeholders
    pub fn placeholder_byte_len(&self) -> usize {
        let tag_len: usize = self.tag_fields.iter()
            .map(|f| f.distributions.max_tag_len())
            .sum();
        let numeric_len = self.numeric_fields.len() * 24; // Max numeric format length
        tag_len + numeric_len
    }
}
```

### CLI Syntax

```bash
# Multiple tag fields
--tag-field "category:electronics:30,clothing:25,home:20"
--tag-field "brand:nike:20,adidas:15,puma:10"

# Or combined
--tag-fields "category:electronics:30|clothing:25,brand:nike:20|adidas:15"
```

### Template Generation

The `create_vec_load_template` function would iterate over all fields:

```rust
fn create_vec_load_template(search_config: &SearchConfig, key_width: usize) -> CommandTemplate {
    let mut template = CommandTemplate::new("HSET")
        .arg_str("HSET")
        .arg_prefixed_key_with_cluster_tag(&search_config.prefix, key_width)
        .arg_str(&search_config.vector_field)
        .arg_vector(search_config.vec_byte_len());

    // Add all tag fields
    for tag_field in &search_config.field_configs.tag_fields {
        template = template
            .arg_str(&tag_field.name)
            .arg_tag_placeholder(tag_field.distributions.max_tag_len());
    }

    // Add all numeric fields
    for numeric_field in &search_config.field_configs.numeric_fields {
        template = template
            .arg_str(&numeric_field.name)
            .arg_numeric_placeholder(numeric_field.max_byte_len());
    }

    template
}
```

---

## 3. Generic Keyspace and Fieldspace Operations

### Current State
- Key = prefix + zero-padded number (e.g., "key:000000000001")
- For HSET: single key with "field" as field name
- No concept of field ranges within keys

### Proposed Design

```rust
/// Keyspace configuration for any data type
#[derive(Debug, Clone)]
pub struct KeyspaceConfig {
    /// Key prefix (e.g., "user:", "session:")
    pub prefix: String,
    /// Start offset for key numbering
    pub start_offset: u64,
    /// Number of keys in keyspace
    pub key_count: u64,
    /// Key number width (zero-padding)
    pub key_width: usize,
    /// How to progress through keys
    pub progression: Progression,
}

/// Fieldspace configuration for hash/json operations
#[derive(Debug, Clone)]
pub struct FieldspaceConfig {
    /// Field name prefix (e.g., "field:", "attr_")
    pub prefix: String,
    /// Number of fields per key
    pub field_count: u64,
    /// Field number width (zero-padding)
    pub field_width: usize,
    /// How to progress through fields
    pub progression: Progression,
    /// Field value configuration
    pub value_config: FieldValueConfig,
}

/// How to progress through a range
#[derive(Debug, Clone, Copy)]
pub enum Progression {
    /// Sequential: 0, 1, 2, 3, ...
    Sequential,
    /// Random uniform selection
    Random,
    /// Zipfian: favors lower indices (hot keys/fields)
    Zipfian { skew: f64 },
    /// Round-robin across workers
    RoundRobin,
}

/// Configuration for field values
#[derive(Debug, Clone)]
pub enum FieldValueConfig {
    /// Fixed-size random bytes
    RandomBytes { size: usize },
    /// Fixed literal value
    Literal(Vec<u8>),
    /// Numeric with distribution
    Numeric(NumericFieldConfig),
    /// Tag values
    Tag(TagDistributionSet),
}
```

### CLI Syntax

```bash
# Hash operations with keyspace and fieldspace
--keyspace "user:0:1000000:12:seq"     # prefix:start:count:width:progression
--fieldspace "attr_:100:3:random"      # prefix:count:width:progression

# Example: HSET user:000000000001 attr_001 value
./benchmark -t hset \
  --keyspace "user:0:1000000:12:seq" \
  --fieldspace "attr_:100:3:random" \
  --data-size 256

# Example: HGET with specific field range
./benchmark -t hget \
  --keyspace "user:0:1000000:12:random" \
  --fieldspace "attr_:100:3:zipfian:0.99"
```

### Implementation Architecture

```rust
/// Extended placeholder types for field operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceholderType {
    // Existing types
    Key,
    ClusterTag,
    Vector,
    QueryVector,
    RandInt,
    Tag,
    Numeric,

    // New types for field operations
    FieldName,      // Field name placeholder (prefix + number)
    FieldValue,     // Field value placeholder
    KeyOffset,      // Key with offset applied
}

/// Context for field-level operations
pub trait FieldContext: Send {
    /// Get next field index for this key
    fn next_field_idx(&self, key_num: u64) -> u64;

    /// Generate field name bytes
    fn fill_field_name(&self, field_idx: u64, buf: &mut [u8]);

    /// Generate field value bytes
    fn fill_field_value(&self, key_num: u64, field_idx: u64, buf: &mut [u8]);

    /// Number of fields per key
    fn fields_per_key(&self) -> u64;
}
```

### Workload Types Extended

```rust
pub enum WorkloadType {
    // ... existing types ...

    // Extended hash operations
    HsetMulti,      // HSET with multiple fields
    HgetField,      // HGET specific field
    HmgetFields,    // HMGET multiple fields
    HdelField,      // HDEL specific field

    // JSON operations (future)
    JsonSet,
    JsonGet,
    JsonMget,
}
```

---

## 4. WorkloadContext Extensions

### Current fill_tag_placeholder

The existing `fill_tag_placeholder` method handles single tag field. We need to extend this for multiple fields.

### Proposed Extension

```rust
pub trait WorkloadContext: Send {
    // ... existing methods ...

    /// Fill a specific tag field placeholder
    fn fill_tag_field(&self, field_idx: usize, key_num: u64, buf: &mut [u8]) {
        buf.fill(b',');
    }

    /// Fill a specific numeric field placeholder
    fn fill_numeric_field(&self, field_idx: usize, key_num: u64, buf: &mut [u8]) {
        buf.fill(b'0');
    }

    /// Get field context for hash/json operations
    fn field_context(&self) -> Option<&dyn FieldContext> {
        None
    }
}
```

### PlaceholderType Extensions

```rust
pub enum PlaceholderType {
    // ... existing ...

    /// Tag field at specific index
    TagField(usize),
    /// Numeric field at specific index
    NumericField(usize),
    /// Field name (for hash/json field operations)
    FieldName,
}
```

---

## 5. Migration Path

### Phase 1: Numeric Distributions (Minimal Breaking Change)
- Add `NumericFieldConfig` and `NumericDistribution`
- Keep backward compatibility: `--numeric-field name` still works
- Extended syntax opt-in: `--numeric-field "name:type:dist:params"`

### Phase 2: Multiple Fields
- Add `FieldConfigs` to `SearchConfig`
- Deprecate single `tag_field`/`numeric_field` (with compat layer)
- Add `--tag-fields` and `--numeric-fields` CLI args

### Phase 3: Generic Keyspace/Fieldspace
- Add `KeyspaceConfig` and `FieldspaceConfig` to `BenchmarkConfig`
- Add `FieldContext` trait
- Extend `PlaceholderType` for field operations
- Add new workload types (HsetMulti, HgetField, etc.)

---

## 6. Example Use Cases

### Vector Search with Rich Metadata
```bash
# Load vectors with category, brand, timestamp, and price
./benchmark -t vec-load \
  --dataset products.bin \
  --tag-fields "category:electronics:30|clothing:25,brand:nike:20|adidas:15" \
  --numeric-fields "timestamp:unix:uniform:1609459200:1704067200,price:float2:normal:50:20"
```

### Hash CRUD Benchmark
```bash
# Write 1M keys with 100 fields each
./benchmark -t hset-multi \
  --keyspace "session:0:1000000:12:seq" \
  --fieldspace "data_:100:3:seq" \
  --data-size 256 \
  -n 100000000

# Read random fields from random keys (simulating cache access)
./benchmark -t hget-field \
  --keyspace "session:0:1000000:12:zipfian:0.99" \
  --fieldspace "data_:100:3:zipfian:0.9" \
  -n 10000000
```

### JSON Path Operations (Future)
```bash
# Set nested JSON fields
./benchmark -t json-set \
  --keyspace "doc:0:100000:8:seq" \
  --json-path "$.attributes[__FIELD_IDX__].value" \
  --fieldspace ":50:2:random" \
  --data-size 128
```

---

## 7. Open Questions

1. **Field value correlation**: Should field values be correlated with key values?
   E.g., timestamp increasing with key number?

2. **Multi-field atomicity**: For HMSET with multiple fields, should all fields
   be set in one command or separate commands?

3. **Read-after-write patterns**: Should we support mixed read/write patterns
   on the same keyspace (e.g., 80% read, 20% write)?

4. **JSON path complexity**: How complex should JSON path expressions be?
   Simple field access vs. nested arrays/objects?

5. **Backward compatibility**: How strictly should we maintain CLI compatibility?
   Deprecation warnings vs. hard breaks?
