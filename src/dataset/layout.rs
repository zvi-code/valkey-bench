//! Layout computation from schema
//!
//! This module computes byte offsets and sizes for fields and sections
//! based on the schema definition. All offsets are computed at load time,
//! enabling O(1) mmap access to any field in any record.

use std::collections::HashMap;

use super::schema::{DatasetSchema, FieldDef, LengthSpec};

/// Computed field layout within a record
#[derive(Debug, Clone)]
pub struct FieldLayout {
    /// Field name
    pub name: String,

    /// Byte offset within record
    pub offset: usize,

    /// Total byte size (including length prefix for variable fields)
    pub size: usize,

    /// Offset to actual data within field (after length prefix if variable)
    pub data_offset: usize,

    /// Whether this is a variable-length field
    pub is_variable: bool,

    /// Whether this is a vector field
    pub is_vector: bool,

    /// Vector dimensions (if vector field)
    pub dimensions: Option<u32>,
}

/// Computed record layout
#[derive(Debug, Clone)]
pub struct RecordLayout {
    /// Field layouts in order
    pub fields: Vec<FieldLayout>,

    /// Map from field name to index
    pub field_by_name: HashMap<String, usize>,

    /// Total record size in bytes
    pub total_size: usize,

    /// Index of first vector field (if any)
    pub first_vector_idx: Option<usize>,
}

impl RecordLayout {
    /// Compute layout from field definitions
    pub fn from_fields(fields: &[FieldDef]) -> Self {
        let mut layout_fields = Vec::with_capacity(fields.len());
        let mut field_by_name = HashMap::new();
        let mut offset = 0;
        let mut first_vector_idx = None;

        for (idx, field) in fields.iter().enumerate() {
            let size = field.byte_size();
            let is_variable = field.length == Some(LengthSpec::Variable);
            let data_offset = if is_variable { 4 } else { 0 };
            let is_vector = field.is_vector();

            if is_vector && first_vector_idx.is_none() {
                first_vector_idx = Some(idx);
            }

            layout_fields.push(FieldLayout {
                name: field.name.clone(),
                offset,
                size,
                data_offset,
                is_variable,
                is_vector,
                dimensions: field.dimensions,
            });

            field_by_name.insert(field.name.clone(), idx);
            offset += size;
        }

        Self {
            fields: layout_fields,
            field_by_name,
            total_size: offset,
            first_vector_idx,
        }
    }

    /// Get field index by name
    #[inline]
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.field_by_name.get(name).copied()
    }

    /// Get field layout by name
    #[inline]
    pub fn field(&self, name: &str) -> Option<&FieldLayout> {
        self.field_index(name).map(|idx| &self.fields[idx])
    }

    /// Get first vector field layout
    #[inline]
    pub fn first_vector_field(&self) -> Option<&FieldLayout> {
        self.first_vector_idx.map(|idx| &self.fields[idx])
    }
}

/// Computed section layout for the binary file
#[derive(Debug, Clone)]
pub struct SectionLayout {
    /// Records section offset (always 0 for schema-driven files)
    pub records_offset: usize,

    /// Records section size in bytes
    pub records_size: usize,

    /// Number of records
    pub record_count: u64,

    /// Record size in bytes
    pub record_size: usize,

    /// Keys section offset (None if keys are generated)
    pub keys_offset: Option<usize>,

    /// Key entry size
    pub keys_entry_size: Option<usize>,

    /// Queries section offset
    pub queries_offset: Option<usize>,

    /// Queries section size
    pub queries_size: Option<usize>,

    /// Query record size (may be smaller than record_size if subset of fields)
    pub query_record_size: usize,

    /// Number of queries
    pub query_count: u64,

    /// Indices of fields included in query records
    pub query_field_indices: Vec<usize>,

    /// Ground truth section offset
    pub ground_truth_offset: Option<usize>,

    /// Number of neighbors per query
    pub neighbors_per_query: usize,

    /// Ground truth ID size (4 for u32, 8 for u64)
    pub gt_id_size: usize,

    /// Total file size
    pub total_size: usize,
}

impl SectionLayout {
    /// Compute section layout from schema and record layout
    pub fn compute(schema: &DatasetSchema, record_layout: &RecordLayout) -> Self {
        let record_count = schema.sections.records.count;
        let record_size = record_layout.total_size;
        let records_size = record_count as usize * record_size;

        let mut next_offset = records_size;

        // Keys section
        let (keys_offset, keys_entry_size) = if schema.sections.keys.present {
            let entry_size = schema.sections.keys.max_bytes.unwrap_or(64) as usize;
            let length = schema.sections.keys.length.unwrap_or_default();
            let actual_size = if length == LengthSpec::Variable {
                4 + entry_size // u32 prefix + data
            } else {
                entry_size
            };
            let offset = next_offset;
            next_offset += record_count as usize * actual_size;
            (Some(offset), Some(actual_size))
        } else {
            (None, None)
        };

        // Compute query field indices and query record size
        let query_field_indices: Vec<usize> =
            if let Some(ref query_fields) = schema.sections.queries.query_fields {
                query_fields
                    .iter()
                    .filter_map(|name| record_layout.field_index(name))
                    .collect()
            } else {
                // Default: all fields
                (0..record_layout.fields.len()).collect()
            };

        let query_record_size: usize = query_field_indices
            .iter()
            .map(|&idx| record_layout.fields[idx].size)
            .sum();

        // Queries section
        let query_count = if schema.sections.queries.present {
            schema.sections.queries.count.unwrap_or(0)
        } else {
            0
        };

        let (queries_offset, queries_size) = if schema.sections.queries.present && query_count > 0 {
            let size = query_count as usize * query_record_size;
            let offset = next_offset;
            next_offset += size;
            (Some(offset), Some(size))
        } else {
            (None, None)
        };

        // Ground truth section
        let neighbors_per_query = schema.neighbors_per_query();
        let gt_id_size = schema.ground_truth_id_size();

        let ground_truth_offset = if schema.has_ground_truth() && query_count > 0 {
            let offset = next_offset;
            next_offset += query_count as usize * neighbors_per_query * gt_id_size;
            Some(offset)
        } else {
            None
        };

        Self {
            records_offset: 0,
            records_size,
            record_count,
            record_size,
            keys_offset,
            keys_entry_size,
            queries_offset,
            queries_size,
            query_record_size,
            query_count,
            query_field_indices,
            ground_truth_offset,
            neighbors_per_query,
            gt_id_size,
            total_size: next_offset,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::schema::{DType, Encoding, FieldType};

    fn make_vector_field(name: &str, dim: u32) -> FieldDef {
        FieldDef {
            name: name.to_string(),
            field_type: FieldType::Vector,
            dtype: Some(DType::Float32),
            dimensions: Some(dim),
            encoding: None,
            length: None,
            max_bytes: None,
        }
    }

    fn make_text_field(name: &str, max_bytes: u32, variable: bool) -> FieldDef {
        FieldDef {
            name: name.to_string(),
            field_type: FieldType::Text,
            dtype: None,
            dimensions: None,
            encoding: Some(Encoding::Utf8),
            length: Some(if variable {
                LengthSpec::Variable
            } else {
                LengthSpec::Fixed
            }),
            max_bytes: Some(max_bytes),
        }
    }

    fn make_numeric_field(name: &str) -> FieldDef {
        FieldDef {
            name: name.to_string(),
            field_type: FieldType::Numeric,
            dtype: Some(DType::Float64),
            dimensions: None,
            encoding: None,
            length: None,
            max_bytes: None,
        }
    }

    #[test]
    fn test_record_layout_single_vector() {
        let fields = vec![make_vector_field("embedding", 128)];
        let layout = RecordLayout::from_fields(&fields);

        assert_eq!(layout.fields.len(), 1);
        assert_eq!(layout.total_size, 128 * 4); // 512 bytes
        assert_eq!(layout.first_vector_idx, Some(0));

        let field = &layout.fields[0];
        assert_eq!(field.name, "embedding");
        assert_eq!(field.offset, 0);
        assert_eq!(field.size, 512);
        assert!(!field.is_variable);
        assert!(field.is_vector);
    }

    #[test]
    fn test_record_layout_multi_field() {
        let fields = vec![
            make_vector_field("embedding", 4),  // 16 bytes
            make_text_field("category", 32, false), // 32 bytes
            make_numeric_field("price"),        // 8 bytes
        ];
        let layout = RecordLayout::from_fields(&fields);

        assert_eq!(layout.fields.len(), 3);
        assert_eq!(layout.total_size, 16 + 32 + 8); // 56 bytes

        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[0].size, 16);

        assert_eq!(layout.fields[1].offset, 16);
        assert_eq!(layout.fields[1].size, 32);

        assert_eq!(layout.fields[2].offset, 48);
        assert_eq!(layout.fields[2].size, 8);

        assert_eq!(layout.field_index("embedding"), Some(0));
        assert_eq!(layout.field_index("category"), Some(1));
        assert_eq!(layout.field_index("price"), Some(2));
        assert_eq!(layout.field_index("nonexistent"), None);
    }

    #[test]
    fn test_record_layout_variable_length() {
        let fields = vec![
            make_text_field("title", 64, false),   // 64 bytes
            make_text_field("content", 1024, true), // 4 + 1024 = 1028 bytes
        ];
        let layout = RecordLayout::from_fields(&fields);

        assert_eq!(layout.total_size, 64 + 1028);

        assert!(!layout.fields[0].is_variable);
        assert_eq!(layout.fields[0].data_offset, 0);

        assert!(layout.fields[1].is_variable);
        assert_eq!(layout.fields[1].data_offset, 4); // After u32 length prefix
    }

    #[test]
    fn test_section_layout() {
        use crate::dataset::schema::DatasetSchema;

        let yaml = r#"
version: 1
record:
  fields:
    - name: embedding
      type: vector
      dtype: float32
      dimensions: 4
sections:
  records:
    count: 100
  keys:
    present: true
    max_bytes: 24
  queries:
    present: true
    count: 10
    query_fields:
      - embedding
  ground_truth:
    present: true
    neighbors_per_query: 5
    id_type: u64
"#;
        let schema = DatasetSchema::from_yaml(yaml).unwrap();
        let record_layout = RecordLayout::from_fields(schema.record.get_fields().unwrap());
        let section_layout = SectionLayout::compute(&schema, &record_layout);

        // Record size: 4 * 4 = 16 bytes
        assert_eq!(section_layout.record_size, 16);

        // Records: 100 * 16 = 1600 bytes
        assert_eq!(section_layout.records_offset, 0);
        assert_eq!(section_layout.records_size, 1600);

        // Keys: 100 * 24 = 2400 bytes at offset 1600
        assert_eq!(section_layout.keys_offset, Some(1600));
        assert_eq!(section_layout.keys_entry_size, Some(24));

        // Queries: 10 * 16 = 160 bytes at offset 4000
        assert_eq!(section_layout.queries_offset, Some(4000));
        assert_eq!(section_layout.query_record_size, 16);

        // Ground truth: 10 * 5 * 8 = 400 bytes at offset 4160
        assert_eq!(section_layout.ground_truth_offset, Some(4160));
        assert_eq!(section_layout.neighbors_per_query, 5);
        assert_eq!(section_layout.gt_id_size, 8);

        // Total: 4160 + 400 = 4560 bytes
        assert_eq!(section_layout.total_size, 4560);
    }
}
