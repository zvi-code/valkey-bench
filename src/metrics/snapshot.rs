//! Cluster snapshot for temporal metrics comparison
//!
//! Captures point-in-time metrics from all cluster nodes and supports
//! diff calculations between snapshots.

use std::collections::HashMap;
use std::time::Instant;

use super::info_fields::{
    calculate_diff, format_value, parse_value, AggregationType, DiffType, DisplayFormat,
    FieldSnapshot, FieldValue, InfoFieldType, NodeFilter,
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

        let mut values = HashMap::new();

        for line in info_lines.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Find matching field definitions
            for field in &self.fields {
                if !field.matches(line.split(':').next().unwrap_or("")) {
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
                    let field_key = format!(
                        "{}{}",
                        field.name,
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
            let field_key = format!(
                "{}{}",
                field.name,
                field
                    .parse_config
                    .key
                    .as_ref()
                    .map(|k| format!(":{}", k))
                    .unwrap_or_default()
            );

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
                    DiffType::RateCount => format!("{:.2}/s", rate),
                    DiffType::RateMicrosec => format!("{:.2}s/s", rate),
                    DiffType::MemoryGrowth => format!("{:.2}MB/s", rate),
                    DiffType::PercentageChange => format!("{:+.2}%", rate),
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
        let field_key = format!(
            "{}{}",
            field.name,
            field
                .parse_config
                .key
                .as_ref()
                .map(|k| format!(":{}", k))
                .unwrap_or_default()
        );

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

            field_diffs.push(FieldDiff {
                field_name: field_key,
                old_value,
                new_value,
                delta,
                rate,
                display_format: field.display_format,
                diff_type: field.diff_type,
            });
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
        let field_def = fields.iter().find(|f| {
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
}
