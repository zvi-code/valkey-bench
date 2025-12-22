//! Cluster snapshot for temporal metrics comparison
//!
//! Captures point-in-time metrics from all cluster nodes and supports
//! diff calculations between snapshots.

use std::collections::HashMap;
use std::time::Instant;

use super::info_fields::{
    calculate_diff, format_value, parse_value, AggregationType, DiffType, DisplayFormat,
    FieldSnapshot, FieldValue, InfoFieldType, MatchStrategy, NodeFilter,
};

/// Snapshot of a single field value from one node
#[derive(Debug, Clone)]
pub struct NodeFieldValue {
    /// Node identifier
    pub node_id: String,
    /// Whether node is primary
    pub is_primary: bool,
    /// Parsed value
    pub value: i64,
    /// Raw string value
    pub raw_value: String,
}

/// Snapshot of cluster state at a point in time
#[derive(Debug, Clone)]
pub struct ClusterSnapshot {
    /// Snapshot label
    pub label: String,
    /// Timestamp when snapshot was taken
    pub timestamp: Instant,
    /// Number of nodes
    pub node_count: usize,
    /// Node identifiers
    pub node_ids: Vec<String>,
    /// Field snapshots (field_name -> snapshot)
    pub fields: HashMap<String, FieldSnapshot>,
}

impl ClusterSnapshot {
    /// Create a new empty snapshot
    pub fn new(label: &str, node_ids: Vec<String>) -> Self {
        let node_count = node_ids.len();
        Self {
            label: label.to_string(),
            timestamp: Instant::now(),
            node_count,
            node_ids,
            fields: HashMap::new(),
        }
    }

    /// Get field value
    pub fn get_field(&self, name: &str) -> Option<&FieldSnapshot> {
        self.fields.get(name)
    }

    /// Get aggregated field value
    pub fn get_value(&self, name: &str) -> Option<i64> {
        self.fields.get(name).filter(|f| f.valid).map(|f| f.value.value)
    }
}

/// Builder for creating cluster snapshots
pub struct SnapshotBuilder {
    label: String,
    node_ids: Vec<String>,
    fields: Vec<InfoFieldType>,
    node_values: Vec<HashMap<String, NodeFieldValue>>,
}

impl SnapshotBuilder {
    /// Create a new snapshot builder
    pub fn new(label: &str, fields: Vec<InfoFieldType>) -> Self {
        Self {
            label: label.to_string(),
            node_ids: Vec::new(),
            fields,
            node_values: Vec::new(),
        }
    }

    /// Add node data from INFO response lines
    pub fn add_node(&mut self, node_id: &str, is_primary: bool, info_lines: &str) {
        self.node_ids.push(node_id.to_string());

        let mut values: HashMap<String, NodeFieldValue> = HashMap::new();

        for line in info_lines.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Find matching field definitions
            for field in &self.fields {
                let field_name_in_line = line.split(':').next().unwrap_or("");
                if !field.matches(field_name_in_line) {
                    continue;
                }

                // Check node filter
                let include = match field.node_filter {
                    NodeFilter::PrimaryOnly => is_primary,
                    NodeFilter::ReplicaOnly => !is_primary,
                    NodeFilter::All => true,
                };

                if !include {
                    continue;
                }

                // Parse value
                if let Some(value) = parse_value(line, &field.parse_config) {
                    // For prefix-matched fields, use the actual field name from the line
                    // (e.g., "cmdstat_get" instead of "cmdstat_") to track each command separately
                    let base_name = if field.match_strategy == MatchStrategy::Prefix {
                        field_name_in_line
                    } else {
                        &field.name
                    };
                    
                    let field_key = format!(
                        "{}{}",
                        base_name,
                        field
                            .parse_config
                            .key
                            .as_ref()
                            .map(|k| format!(":{}", k))
                            .unwrap_or_default()
                    );

                    values.insert(
                        field_key,
                        NodeFieldValue {
                            node_id: node_id.to_string(),
                            is_primary,
                            value,
                            raw_value: line.to_string(),
                        },
                    );
                }
            }
        }

        self.node_values.push(values);
    }

    /// Build the final snapshot
    pub fn build(self) -> ClusterSnapshot {
        let mut snapshot = ClusterSnapshot::new(&self.label, self.node_ids.clone());

        // Aggregate each field
        for field in &self.fields {
            // For prefix-matched fields, find all unique keys that match the pattern
            let field_keys: Vec<String> = if field.match_strategy == MatchStrategy::Prefix {
                // Collect all unique keys that start with the pattern and have the right suffix
                let suffix = field
                    .parse_config
                    .key
                    .as_ref()
                    .map(|k| format!(":{}", k))
                    .unwrap_or_default();
                
                let mut keys: Vec<String> = self.node_values
                    .iter()
                    .flat_map(|nv| nv.keys())
                    .filter(|k| {
                        let name_lower = field.name.to_ascii_lowercase();
                        let k_lower = k.to_ascii_lowercase();
                        k_lower.starts_with(&name_lower) && k.ends_with(&suffix)
                    })
                    .cloned()
                    .collect();
                keys.sort();
                keys.dedup();
                keys
            } else {
                // Exact match - single key
                vec![format!(
                    "{}{}",
                    field.name,
                    field
                        .parse_config
                        .key
                        .as_ref()
                        .map(|k| format!(":{}", k))
                        .unwrap_or_default()
                )]
            };

            // Process each field key
            for field_key in field_keys {
                let mut values: Vec<i64> = Vec::new();
                let mut per_node: Vec<(String, i64)> = Vec::new();
                let mut node_count = 0;

                for (idx, node_values) in self.node_values.iter().enumerate() {
                    if let Some(nv) = node_values.get(&field_key) {
                        values.push(nv.value);
                        per_node.push((self.node_ids[idx].clone(), nv.value));
                        node_count += 1;
                    }
                }

                if values.is_empty() {
                    continue;
                }

                // Aggregate based on type
                let aggregated = match field.aggregation_type {
                    AggregationType::Sum => FieldValue::new(values.iter().sum()),
                    AggregationType::Average => {
                        let sum: i64 = values.iter().sum();
                        FieldValue::new(sum / values.len() as i64)
                    }
                    AggregationType::Max => FieldValue::new(*values.iter().max().unwrap_or(&0)),
                    AggregationType::MinMax => {
                        let min = *values.iter().min().unwrap_or(&0);
                        let max = *values.iter().max().unwrap_or(&0);
                        FieldValue::with_minmax(min, max)
                    }
                };

                snapshot.fields.insert(
                    field_key.clone(),
                    FieldSnapshot {
                        field_name: field_key,
                        value: aggregated,
                        per_node_values: if field.track_per_node {
                            Some(per_node)
                        } else {
                            None
                        },
                        node_count,
                        valid: true,
                    },
                );
            }
        }

        snapshot
    }
}

/// Diff between two snapshots
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    /// Old snapshot label
    pub old_label: String,
    /// New snapshot label
    pub new_label: String,
    /// Time between snapshots
    pub elapsed_secs: f64,
    /// Field diffs
    pub fields: Vec<FieldDiff>,
}

/// Diff for a single field
#[derive(Debug, Clone)]
pub struct FieldDiff {
    /// Field name
    pub field_name: String,
    /// Old value
    pub old_value: i64,
    /// New value
    pub new_value: i64,
    /// Absolute delta
    pub delta: i64,
    /// Rate (if applicable)
    pub rate: Option<f64>,
    /// Display format
    pub display_format: DisplayFormat,
    /// Diff type used
    pub diff_type: DiffType,
    /// Per-node deltas: (node_id, old_value, new_value, delta)
    pub per_node_deltas: Option<Vec<(String, i64, i64, i64)>>,
}

/// Per-node delta for a specific field
#[derive(Debug, Clone)]
pub struct NodeDelta {
    pub node_id: String,
    pub old_value: i64,
    pub new_value: i64,
    pub delta: i64,
    pub rate: Option<f64>,
}

impl FieldDiff {
    /// Format the diff for display
    pub fn format(&self) -> String {
        let new_formatted = format_value(
            &FieldValue::new(self.new_value),
            self.display_format,
        );

        match self.rate {
            Some(rate) => {
                let rate_str = match self.diff_type {
                    DiffType::RateCount => format!("{:.1}/s", rate),
                    DiffType::RateMicrosec => format!("{:.1}s/s", rate),
                    DiffType::MemoryGrowth => format!("{:.0}MB/s", rate),
                    DiffType::PercentageChange => format!("{:+.1}%", rate),
                    DiffType::None => String::new(),
                };
                format!("{} ({})", new_formatted, rate_str)
            }
            None => new_formatted,
        }
    }
}

/// Compare two snapshots and generate diff
pub fn compare_snapshots(
    old: &ClusterSnapshot,
    new: &ClusterSnapshot,
    fields: &[InfoFieldType],
) -> SnapshotDiff {
    let elapsed_secs = new.timestamp.duration_since(old.timestamp).as_secs_f64();

    let mut field_diffs = Vec::new();

    for field in fields {
        // For prefix-matched fields, find all keys that match in both snapshots
        let field_keys: Vec<String> = if field.match_strategy == MatchStrategy::Prefix {
            let suffix = field
                .parse_config
                .key
                .as_ref()
                .map(|k| format!(":{}", k))
                .unwrap_or_default();
            
            // Collect keys from both old and new snapshots
            let mut keys: Vec<String> = old.fields.keys()
                .chain(new.fields.keys())
                .filter(|k| {
                    let name_lower = field.name.to_ascii_lowercase();
                    let k_lower = k.to_ascii_lowercase();
                    k_lower.starts_with(&name_lower) && k.ends_with(&suffix)
                })
                .cloned()
                .collect();
            keys.sort();
            keys.dedup();
            keys
        } else {
            vec![format!(
                "{}{}",
                field.name,
                field
                    .parse_config
                    .key
                    .as_ref()
                    .map(|k| format!(":{}", k))
                    .unwrap_or_default()
            )]
        };

        for field_key in field_keys {
            let old_field = old.fields.get(&field_key);
            let new_field = new.fields.get(&field_key);

            if let (Some(old_f), Some(new_f)) = (old_field, new_field) {
                if !old_f.valid || !new_f.valid {
                    continue;
                }

                let old_value = old_f.value.value;
                let new_value = new_f.value.value;
                let delta = new_value - old_value;

                let rate = if field.diff_type != DiffType::None {
                    calculate_diff(old_value, new_value, elapsed_secs, field.diff_type)
                } else {
                    None
                };

                // Compute per-node deltas if both snapshots have per-node values
                let per_node_deltas = if field.track_per_node {
                    match (&old_f.per_node_values, &new_f.per_node_values) {
                        (Some(old_nodes), Some(new_nodes)) => {
                            let mut deltas = Vec::new();
                            // Build map of old values
                            let old_map: std::collections::HashMap<&str, i64> = old_nodes
                                .iter()
                                .map(|(k, v)| (k.as_str(), *v))
                                .collect();

                            for (node_id, new_val) in new_nodes {
                                let old_val = *old_map.get(node_id.as_str()).unwrap_or(&0);
                                deltas.push((node_id.clone(), old_val, *new_val, *new_val - old_val));
                            }
                            Some(deltas)
                        }
                        _ => None,
                    }
                } else {
                    None
                };

                field_diffs.push(FieldDiff {
                    field_name: field_key,
                    old_value,
                    new_value,
                    delta,
                    rate,
                    display_format: field.display_format,
                    diff_type: field.diff_type,
                    per_node_deltas,
                });
            }
        }
    }

    SnapshotDiff {
        old_label: old.label.clone(),
        new_label: new.label.clone(),
        elapsed_secs,
        fields: field_diffs,
    }
}

/// Print snapshot diff in a formatted table
pub fn print_snapshot_diff(diff: &SnapshotDiff, fields: &[InfoFieldType]) {
    println!(
        "\n=== Snapshot Diff: {} -> {} ({:.2}s) ===\n",
        diff.old_label, diff.new_label, diff.elapsed_secs
    );

    // Group fields by category (based on prefix)
    let mut current_prefix = String::new();

    for field_diff in &diff.fields {
        // Find field definition
        let _field_def = fields.iter().find(|f| {
            let key = format!(
                "{}{}",
                f.name,
                f.parse_config
                    .key
                    .as_ref()
                    .map(|k| format!(":{}", k))
                    .unwrap_or_default()
            );
            key == field_diff.field_name
        });

        // Print section header if prefix changed
        let prefix = field_diff
            .field_name
            .split('_')
            .take(2)
            .collect::<Vec<_>>()
            .join("_");
        if prefix != current_prefix {
            println!("\n[{}]", prefix);
            current_prefix = prefix;
        }

        // Skip fields with no diff
        if field_diff.diff_type == DiffType::None && field_diff.delta == 0 {
            continue;
        }

        println!(
            "  {:50} {}",
            field_diff.field_name,
            field_diff.format()
        );
    }

    println!();
}

/// Print per-node diff table for selected fields
///
/// Shows how load/metrics are distributed across nodes in a cluster.
/// Useful for identifying:
/// - Uneven load distribution
/// - Node-specific issues
/// - Client distribution problems
pub fn print_per_node_diff(diff: &SnapshotDiff, field_names: &[&str], elapsed_secs: f64) {
    // Collect fields that have per-node data
    let fields_with_nodes: Vec<&FieldDiff> = diff
        .fields
        .iter()
        .filter(|f| {
            field_names.iter().any(|n| f.field_name.contains(n))
                && f.per_node_deltas.is_some()
                && f.per_node_deltas.as_ref().map(|d| !d.is_empty()).unwrap_or(false)
        })
        .collect();

    if fields_with_nodes.is_empty() {
        println!("\n(No per-node data available for selected fields)");
        return;
    }

    println!("\n=== Per-Node Distribution ({:.2}s) ===\n", elapsed_secs);

    for field in fields_with_nodes {
        if let Some(ref node_deltas) = field.per_node_deltas {
            println!("{}:", field.field_name);
            println!(
                "  {:40} {:>12} {:>12} {:>12} {:>12}",
                "Node", "Before", "After", "Delta", "Rate/s"
            );
            println!("  {}", "-".repeat(88));

            let total_delta: i64 = node_deltas.iter().map(|(_, _, _, d)| d).sum();

            for (node_id, old_val, new_val, delta) in node_deltas {
                let rate = if elapsed_secs > 0.0 {
                    *delta as f64 / elapsed_secs
                } else {
                    0.0
                };
                let pct = if total_delta > 0 {
                    (*delta as f64 / total_delta as f64) * 100.0
                } else {
                    0.0
                };
                println!(
                    "  {:40} {:>12} {:>12} {:>+12} {:>11.1} ({:>5.1}%)",
                    node_id, old_val, new_val, delta, rate, pct
                );
            }

            // Print total row
            let total_old: i64 = node_deltas.iter().map(|(_, o, _, _)| o).sum();
            let total_new: i64 = node_deltas.iter().map(|(_, _, n, _)| n).sum();
            let total_rate = if elapsed_secs > 0.0 {
                total_delta as f64 / elapsed_secs
            } else {
                0.0
            };
            println!("  {}", "-".repeat(88));
            println!(
                "  {:40} {:>12} {:>12} {:>+12} {:>11.1}",
                "TOTAL", total_old, total_new, total_delta, total_rate
            );
            println!();
        }
    }
}

/// Print per-node diff table for all tracked fields with significant deltas
pub fn print_per_node_diff_all(diff: &SnapshotDiff) {
    let elapsed_secs = diff.elapsed_secs;

    // Collect fields that have per-node data with non-zero deltas
    let fields_with_nodes: Vec<&FieldDiff> = diff
        .fields
        .iter()
        .filter(|f| {
            if let Some(ref deltas) = f.per_node_deltas {
                deltas.iter().any(|(_, _, _, d)| *d != 0)
            } else {
                false
            }
        })
        .collect();

    if fields_with_nodes.is_empty() {
        return;
    }

    println!("\n=== Per-Node Distribution ({:.2}s) ===", elapsed_secs);

    for field in fields_with_nodes {
        if let Some(ref node_deltas) = field.per_node_deltas {
            // Skip if all deltas are zero
            if node_deltas.iter().all(|(_, _, _, d)| *d == 0) {
                continue;
            }

            println!("\n{}:", field.field_name);
            println!(
                "  {:40} {:>12} {:>12} {:>12}",
                "Node", "Delta", "Rate/s", "Share"
            );
            println!("  {}", "-".repeat(78));

            let total_delta: i64 = node_deltas.iter().map(|(_, _, _, d)| *d).sum();

            for (node_id, _old_val, _new_val, delta) in node_deltas {
                if *delta == 0 {
                    continue;
                }
                let rate = if elapsed_secs > 0.0 {
                    *delta as f64 / elapsed_secs
                } else {
                    0.0
                };
                let pct = if total_delta != 0 {
                    (*delta as f64 / total_delta as f64) * 100.0
                } else {
                    0.0
                };
                println!(
                    "  {:40} {:>+12} {:>12.1} {:>11.1}%",
                    node_id, delta, rate, pct
                );
            }

            // Print total row
            let total_rate = if elapsed_secs > 0.0 {
                total_delta as f64 / elapsed_secs
            } else {
                0.0
            };
            println!("  {}", "-".repeat(78));
            println!(
                "  {:40} {:>+12} {:>12.1}",
                "TOTAL", total_delta, total_rate
            );
        }
    }
    println!();
}

/// Print per-node diff in matrix format (nodes as columns, like C code)
///
/// Format:
/// ```text
/// ┌─────────────────────────────────────────────────────────────────────────────────┐
/// │                        CLUSTER STATISTICS DELTA                                  │
/// ├─────────────────────────────────────────────────────────────────────────────────┤
/// │ Time Interval: 5.23 sec | Nodes: 3                                              │
/// └─────────────────────────────────────────────────────────────────────────────────┘
///
/// │ METRIC              │ CLUSTER │       │  1-1-P  │       │     │  1-2-R  │       │     │
/// │                     │   Δ     │  /s   │    Δ    │  /s   │  %  │    Δ    │  /s   │  %  │
/// ├─────────────────────┼─────────┼───────┼─────────┼───────┼─────┼─────────┼───────┼─────┤
/// │ search_requests     │  +1000  │  190  │   +250  │   48  │ 25  │   +250  │   48  │ 25  │
/// ```
///
/// Node headers show shard-based names (e.g., "1-1-P" = Shard 1, Index 1, Primary)
/// when topology is provided, otherwise falls back to truncated IP addresses.
pub fn print_per_node_matrix(diff: &SnapshotDiff, topology: Option<&crate::cluster::ClusterTopology>) {
    use tabled::{
        builder::Builder,
        settings::{
            Style, Alignment, Modify, Span,
            object::{Columns, Cell, Rows},
            themes::BorderCorrection,
        },
    };
    
    let elapsed_secs = diff.elapsed_secs;
    if elapsed_secs <= 0.0 {
        return;
    }

    // Collect fields that have per-node data with non-zero deltas
    let fields_with_nodes: Vec<&FieldDiff> = diff
        .fields
        .iter()
        .filter(|f| {
            if let Some(ref deltas) = f.per_node_deltas {
                deltas.iter().any(|(_, _, _, d)| *d != 0)
            } else {
                false
            }
        })
        .collect();

    if fields_with_nodes.is_empty() {
        return;
    }

    // Collect unique node IDs from all fields (preserving order)
    let mut node_ids: Vec<String> = Vec::new();
    for field in &fields_with_nodes {
        if let Some(ref deltas) = field.per_node_deltas {
            for (node_id, _, _, _) in deltas {
                if !node_ids.contains(node_id) {
                    node_ids.push(node_id.clone());
                }
            }
        }
    }

    if node_ids.is_empty() {
        return;
    }

    // Sort node IDs by shard info if topology is available
    if let Some(topo) = topology {
        node_ids.sort_by(|a, b| {
            let node_a = topo.get_node_by_address(
                a.rsplit_once(':').map(|(h, _)| h).unwrap_or(a),
                a.rsplit_once(':').and_then(|(_, p)| p.parse().ok()).unwrap_or(0)
            );
            let node_b = topo.get_node_by_address(
                b.rsplit_once(':').map(|(h, _)| h).unwrap_or(b),
                b.rsplit_once(':').and_then(|(_, p)| p.parse().ok()).unwrap_or(0)
            );
            match (node_a, node_b) {
                (Some(a), Some(b)) => {
                    // Sort by shard_id, then by shard_index
                    let shard_cmp = a.shard_id.cmp(&b.shard_id);
                    if shard_cmp == std::cmp::Ordering::Equal {
                        a.shard_index.cmp(&b.shard_index)
                    } else {
                        shard_cmp
                    }
                }
                _ => a.cmp(b)
            }
        });
    }

    // Generate node display names using topology
    let node_names: Vec<String> = node_ids
        .iter()
        .map(|id| {
            if let Some(topo) = topology {
                topo.get_node_display_name(id)
            } else {
                crate::cluster::truncate_node_address(id)
            }
        })
        .collect();

    // Print title
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                        CLUSTER STATISTICS DELTA                              ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════╣");
    println!("║ Time Interval: {:6.2} sec | Nodes: {:3}                                       ║", elapsed_secs, node_ids.len());
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    // Build table using Builder
    let mut builder = Builder::default();

    // Row 0: Main header with node names (will span 3 columns each)
    let mut header1: Vec<String> = vec!["METRIC".to_string(), "CLUSTER".to_string(), "".to_string()];
    for name in &node_names {
        header1.push(name.clone());
        header1.push("".to_string());
        header1.push("".to_string());
    }
    builder.push_record(header1);

    // Row 1: Sub-header with Δ, /s, % labels
    let mut header2: Vec<String> = vec!["".to_string(), "Δ".to_string(), "/s".to_string()];
    for _ in &node_ids {
        header2.push("Δ".to_string());
        header2.push("/s".to_string());
        header2.push("%".to_string());
    }
    builder.push_record(header2);

    // Build data rows
    for field in &fields_with_nodes {
        if let Some(ref node_deltas) = field.per_node_deltas {
            // Skip if all deltas are zero
            if node_deltas.iter().all(|(_, _, _, d)| *d == 0) {
                continue;
            }

            let total_delta: i64 = node_deltas.iter().map(|(_, _, _, d)| *d).sum();
            let total_rate = total_delta as f64 / elapsed_secs;

            // Build node delta map
            let node_delta_map: std::collections::HashMap<&str, (i64, i64, i64)> = node_deltas
                .iter()
                .map(|(id, old, new, delta)| (id.as_str(), (*old, *new, *delta)))
                .collect();

            // Build row
            let mut row: Vec<String> = vec![
                field.field_name.clone(),
                format_rate(total_delta as f64),
                format_rate(total_rate),
            ];

            // Add per-node values
            for node_id in &node_ids {
                if let Some((_, _, delta)) = node_delta_map.get(node_id.as_str()) {
                    let rate = *delta as f64 / elapsed_secs;
                    let share = if total_delta != 0 {
                        (*delta as f64 / total_delta as f64) * 100.0
                    } else {
                        0.0
                    };
                    if *delta != 0 {
                        row.push(format_rate(*delta as f64));
                        row.push(format_rate(rate));
                        row.push(format!("{:.0}", share));
                    } else {
                        row.push("-".to_string());
                        row.push("-".to_string());
                        row.push("-".to_string());
                    }
                } else {
                    row.push("-".to_string());
                    row.push("-".to_string());
                    row.push("-".to_string());
                }
            }

            builder.push_record(row);
        }
    }

    let mut table = builder.build();
    
    // Apply column spans FIRST (before styling)
    // CLUSTER spans columns 1-2 (row 0)
    table.with(Modify::new(Cell::new(0, 1)).with(Span::column(2)));
    
    // Each node name spans 3 columns starting at column 3, 6, 9, etc.
    for (i, _) in node_ids.iter().enumerate() {
        let col = 3 + i * 3;
        table.with(Modify::new(Cell::new(0, col)).with(Span::column(3)));
    }

    // Apply styling after spans
    table
        .with(Style::sharp())
        .with(BorderCorrection::span());  // Fix borders after spanning
    
    // Alignment: 
    // - First column (metric names) always left-aligned
    // - Node name headers centered
    // - Sub-headers (Δ, /s, %) centered  
    // - Data values right-aligned
    table.with(Modify::new(Columns::first()).with(Alignment::left()));
    table.with(Modify::new(Columns::new(1..)).with(Alignment::right()));
    // Center the header rows (node names and sub-column labels)
    for col in 1..(3 + node_ids.len() * 3) {
        table.with(Modify::new(Cell::new(0, col)).with(Alignment::center()));
        table.with(Modify::new(Cell::new(1, col)).with(Alignment::center()));
    }

    println!("{}", table);
    println!();
}

/// Format rate value with compact notation
fn format_rate(rate: f64) -> String {
    let abs_rate = rate.abs();
    if abs_rate >= 1_000_000_000.0 {
        format!("{:.0}G", rate / 1_000_000_000.0)
    } else if abs_rate >= 1_000_000.0 {
        format!("{:.0}M", rate / 1_000_000.0)
    } else if abs_rate >= 10_000.0 {
        format!("{:.0}K", rate / 1_000.0)
    } else if abs_rate >= 100.0 {
        format!("{:.0}", rate)
    } else if abs_rate >= 10.0 {
        format!("{:.1}", rate)
    } else {
        format!("{:.2}", rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::info_fields::{default_search_info_fields, ParseConfig};

    #[test]
    fn test_snapshot_builder_basic() {
        let fields = vec![
            InfoFieldType::new("test_counter")
                .aggregate(AggregationType::Sum)
                .diff(DiffType::RateCount),
        ];

        let mut builder = SnapshotBuilder::new("test", fields);
        builder.add_node("node1", true, "test_counter:100\n");
        builder.add_node("node2", true, "test_counter:200\n");

        let snapshot = builder.build();

        assert_eq!(snapshot.node_count, 2);
        assert_eq!(snapshot.get_value("test_counter"), Some(300)); // Sum
    }

    #[test]
    fn test_snapshot_diff() {
        let fields = vec![
            InfoFieldType::new("requests")
                .aggregate(AggregationType::Sum)
                .diff(DiffType::RateCount),
        ];

        let mut builder1 = SnapshotBuilder::new("before", fields.clone());
        builder1.add_node("node1", true, "requests:100\n");
        let snap1 = builder1.build();

        // Simulate time passing
        std::thread::sleep(std::time::Duration::from_millis(100));

        let mut builder2 = SnapshotBuilder::new("after", fields.clone());
        builder2.add_node("node1", true, "requests:200\n");
        let snap2 = builder2.build();

        let diff = compare_snapshots(&snap1, &snap2, &fields);

        assert_eq!(diff.fields.len(), 1);
        assert_eq!(diff.fields[0].delta, 100);
        assert!(diff.fields[0].rate.is_some());
    }

    #[test]
    fn test_node_filter() {
        let fields = vec![
            InfoFieldType::new("primary_only")
                .aggregate(AggregationType::Sum)
                .from_primary_only(),
            InfoFieldType::new("all_nodes").aggregate(AggregationType::Sum),
        ];

        let mut builder = SnapshotBuilder::new("test", fields);
        builder.add_node("primary", true, "primary_only:100\nall_nodes:50\n");
        builder.add_node("replica", false, "primary_only:100\nall_nodes:50\n");

        let snapshot = builder.build();

        // primary_only should only have value from primary
        assert_eq!(snapshot.get_value("primary_only"), Some(100));
        // all_nodes should have sum from both
        assert_eq!(snapshot.get_value("all_nodes"), Some(100));
    }

    #[test]
    fn test_per_node_matrix_formatting() {
        // Create fields with per-node tracking
        let fields = vec![
            InfoFieldType::new("search_requests")
                .aggregate(AggregationType::Sum)
                .diff(DiffType::RateCount),
            InfoFieldType::new("search_cpu_time")
                .aggregate(AggregationType::Sum)
                .diff(DiffType::RateCount),
            InfoFieldType::new("search_index_scans")
                .aggregate(AggregationType::Sum)
                .diff(DiffType::RateCount),
        ];

        // Create "before" snapshot with 3 nodes
        let mut builder1 = SnapshotBuilder::new("before", fields.clone());
        builder1.add_node("192.168.1.10:6379", true, "search_requests:1000\nsearch_cpu_time:500\nsearch_index_scans:2000\n");
        builder1.add_node("192.168.1.11:6379", true, "search_requests:800\nsearch_cpu_time:400\nsearch_index_scans:1600\n");
        builder1.add_node("192.168.1.12:6379", true, "search_requests:1200\nsearch_cpu_time:600\nsearch_index_scans:2400\n");
        let snap1 = builder1.build();

        // Create "after" snapshot (simulating 5 seconds of activity)
        std::thread::sleep(std::time::Duration::from_millis(100));
        let mut builder2 = SnapshotBuilder::new("after", fields.clone());
        builder2.add_node("192.168.1.10:6379", true, "search_requests:1500\nsearch_cpu_time:750\nsearch_index_scans:3000\n");
        builder2.add_node("192.168.1.11:6379", true, "search_requests:1200\nsearch_cpu_time:600\nsearch_index_scans:2400\n");
        builder2.add_node("192.168.1.12:6379", true, "search_requests:2000\nsearch_cpu_time:1000\nsearch_index_scans:4000\n");
        let snap2 = builder2.build();

        let diff = compare_snapshots(&snap1, &snap2, &fields);

        // Print the matrix - this will show in test output with --nocapture
        println!("\n=== Test: Per-Node Matrix Formatting ===");
        print_per_node_matrix(&diff, None);

        // Verify the diff contains expected values
        assert_eq!(diff.fields.len(), 3);
        
        // Total deltas should be: requests=1700, cpu_time=850, index_scans=3400
        let requests_diff = diff.fields.iter().find(|f| f.field_name == "search_requests").unwrap();
        assert_eq!(requests_diff.delta, 1700);
        
        // Check per-node deltas exist
        assert!(requests_diff.per_node_deltas.is_some());
        let per_node = requests_diff.per_node_deltas.as_ref().unwrap();
        assert_eq!(per_node.len(), 3);
    }
}
