//! Vector search operations (FT.CREATE, FT.SEARCH, FT.DROPINDEX)
//!
//! This module provides higher-level operations for managing vector search indices
//! and executing search queries.

use crate::client::RawConnection;
use crate::config::SearchConfig;
use crate::utils::{RespEncoder, RespValue};

/// Create a vector search index using FT.CREATE
///
/// # Arguments
/// * `conn` - Connection to execute command on
/// * `config` - Search configuration with index parameters
/// * `overwrite` - If true, drop existing index first
pub fn create_index(
    conn: &mut RawConnection,
    config: &SearchConfig,
    overwrite: bool,
) -> Result<(), String> {
    // Drop existing index if requested
    if overwrite {
        let _ = drop_index(conn, &config.index_name);
    }

    // Build FT.CREATE command
    let mut encoder = RespEncoder::with_capacity(1024);

    // FT.CREATE index_name ON HASH PREFIX 1 prefix: SCHEMA field_name VECTOR algorithm TYPE FLOAT32 DIM dim DISTANCE_METRIC metric
    let mut args: Vec<&[u8]> = Vec::with_capacity(30);

    args.push(b"FT.CREATE");
    let index_bytes = config.index_name.as_bytes();
    args.push(index_bytes);
    args.push(b"ON");
    args.push(b"HASH");
    args.push(b"PREFIX");
    args.push(b"1");
    let prefix_bytes = config.prefix.as_bytes();
    args.push(prefix_bytes);
    args.push(b"SCHEMA");
    let field_bytes = config.vector_field.as_bytes();
    args.push(field_bytes);
    args.push(b"VECTOR");

    // Algorithm-specific parameters
    let algo_str = match config.algorithm {
        crate::config::VectorAlgorithm::Hnsw => "HNSW",
        crate::config::VectorAlgorithm::Flat => "FLAT",
    };
    args.push(algo_str.as_bytes());

    // Calculate number of attribute pairs (each pair = 2 args)
    // Base: TYPE + DIM + DISTANCE_METRIC = 3 pairs = 6 args
    // Optional HNSW params: EF_CONSTRUCTION (2 args), M (2 args)
    let mut num_attrs = 6; // TYPE + value + DIM + value + DISTANCE_METRIC + value
    if matches!(config.algorithm, crate::config::VectorAlgorithm::Hnsw) {
        if config.ef_construction.is_some() {
            num_attrs += 2;
        }
        if config.hnsw_m.is_some() {
            num_attrs += 2;
        }
    }
    let num_attrs_str = num_attrs.to_string();
    args.push(num_attrs_str.as_bytes());

    // Common vector params
    args.push(b"TYPE");
    args.push(b"FLOAT32");
    args.push(b"DIM");
    let dim_str = config.dim.to_string();
    args.push(dim_str.as_bytes());
    args.push(b"DISTANCE_METRIC");
    let metric_str = match config.distance_metric {
        crate::config::DistanceMetric::L2 => "L2",
        crate::config::DistanceMetric::InnerProduct => "IP",
        crate::config::DistanceMetric::Cosine => "COSINE",
    };
    args.push(metric_str.as_bytes());

    // HNSW-specific parameters
    let ef_str: String;
    let m_str: String;
    if matches!(config.algorithm, crate::config::VectorAlgorithm::Hnsw) {
        if let Some(ef) = config.ef_construction {
            args.push(b"EF_CONSTRUCTION");
            ef_str = ef.to_string();
            args.push(ef_str.as_bytes());
        }
        if let Some(m) = config.hnsw_m {
            args.push(b"M");
            m_str = m.to_string();
            args.push(m_str.as_bytes());
        }
    }

    encoder.encode_command(&args);

    // Execute
    match conn.execute(&encoder) {
        Ok(RespValue::SimpleString(s)) if s == "OK" => Ok(()),
        Ok(RespValue::Error(e)) => Err(e),
        Ok(other) => Err(format!("Unexpected response: {:?}", other)),
        Err(e) => Err(format!("IO error: {}", e)),
    }
}

/// Drop a vector search index
pub fn drop_index(conn: &mut RawConnection, index_name: &str) -> Result<(), String> {
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["FT.DROPINDEX", index_name]);

    match conn.execute(&encoder) {
        Ok(RespValue::SimpleString(s)) if s == "OK" => Ok(()),
        Ok(RespValue::Error(e)) => {
            // Ignore "Unknown Index name" errors
            if e.contains("Unknown Index name") || e.contains("Unknown index name") {
                Ok(())
            } else {
                Err(e)
            }
        }
        Ok(_) => Ok(()),
        Err(e) => Err(format!("IO error: {}", e)),
    }
}

/// Get index information using FT.INFO
pub fn get_index_info(conn: &mut RawConnection, index_name: &str) -> Result<IndexInfo, String> {
    let mut encoder = RespEncoder::with_capacity(64);
    encoder.encode_command_str(&["FT.INFO", index_name]);

    match conn.execute(&encoder) {
        Ok(RespValue::Array(arr)) => parse_index_info(&arr),
        Ok(RespValue::Error(e)) => Err(e),
        Ok(other) => Err(format!("Unexpected response: {:?}", other)),
        Err(e) => Err(format!("IO error: {}", e)),
    }
}

/// Parsed index information
#[derive(Debug, Default)]
pub struct IndexInfo {
    pub index_name: String,
    pub num_docs: u64,
    pub max_doc_id: u64,
    pub num_records: u64,
    pub indexing: bool,
    pub percent_indexed: f64,
}

fn parse_index_info(arr: &[RespValue]) -> Result<IndexInfo, String> {
    let mut info = IndexInfo::default();

    // FT.INFO returns an array of key-value pairs
    let mut i = 0;
    while i < arr.len() - 1 {
        if let RespValue::BulkString(key) = &arr[i] {
            let key_str = String::from_utf8_lossy(key);
            match key_str.as_ref() {
                "index_name" => {
                    if let RespValue::BulkString(v) = &arr[i + 1] {
                        info.index_name = String::from_utf8_lossy(v).to_string();
                    }
                }
                "num_docs" => {
                    if let RespValue::BulkString(v) = &arr[i + 1] {
                        info.num_docs = String::from_utf8_lossy(v).parse().unwrap_or(0);
                    }
                    if let RespValue::Integer(v) = &arr[i + 1] {
                        info.num_docs = *v as u64;
                    }
                }
                "max_doc_id" => {
                    if let RespValue::BulkString(v) = &arr[i + 1] {
                        info.max_doc_id = String::from_utf8_lossy(v).parse().unwrap_or(0);
                    }
                    if let RespValue::Integer(v) = &arr[i + 1] {
                        info.max_doc_id = *v as u64;
                    }
                }
                "num_records" => {
                    if let RespValue::BulkString(v) = &arr[i + 1] {
                        info.num_records = String::from_utf8_lossy(v).parse().unwrap_or(0);
                    }
                    if let RespValue::Integer(v) = &arr[i + 1] {
                        info.num_records = *v as u64;
                    }
                }
                "indexing" => {
                    if let RespValue::BulkString(v) = &arr[i + 1] {
                        info.indexing = String::from_utf8_lossy(v) == "1";
                    }
                    if let RespValue::Integer(v) = &arr[i + 1] {
                        info.indexing = *v == 1;
                    }
                }
                "percent_indexed" => {
                    if let RespValue::BulkString(v) = &arr[i + 1] {
                        info.percent_indexed = String::from_utf8_lossy(v).parse().unwrap_or(0.0);
                    }
                }
                _ => {}
            }
        }
        i += 2;
    }

    Ok(info)
}

/// Wait for index to be fully indexed (background indexing complete)
pub fn wait_for_indexing(
    conn: &mut RawConnection,
    index_name: &str,
    timeout_secs: u64,
) -> Result<(), String> {
    use std::time::{Duration, Instant};

    let start = Instant::now();
    let timeout = Duration::from_secs(timeout_secs);

    loop {
        let info = get_index_info(conn, index_name)?;

        if !info.indexing || info.percent_indexed >= 1.0 {
            return Ok(());
        }

        if start.elapsed() > timeout {
            return Err(format!(
                "Timeout waiting for index {} to complete ({}% indexed)",
                index_name,
                info.percent_indexed * 100.0
            ));
        }

        std::thread::sleep(Duration::from_millis(500));
    }
}

/// Parse FT.SEARCH response to extract document IDs
pub fn parse_search_response(response: &RespValue) -> Vec<String> {
    let mut doc_ids = Vec::new();

    if let RespValue::Array(arr) = response {
        // First element is total count, then alternating doc_id and fields
        let mut i = 1;
        while i < arr.len() {
            if let RespValue::BulkString(doc_id) = &arr[i] {
                doc_ids.push(String::from_utf8_lossy(doc_id).to_string());
            }
            // Skip to next doc_id (skip over fields array if present)
            i += 2;
        }
    }

    doc_ids
}

/// Extract numeric IDs from document keys (e.g., "vec:000000000123" -> 123)
pub fn extract_numeric_ids(doc_ids: &[String], prefix: &str) -> Vec<u64> {
    doc_ids
        .iter()
        .filter_map(|id| {
            let stripped = id.strip_prefix(prefix)?;
            // Handle the case where ID is 0 (all zeros become empty string)
            let trimmed = stripped.trim_start_matches('0');
            if trimmed.is_empty() {
                Some(0)
            } else {
                trimmed.parse().ok()
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_numeric_ids() {
        let doc_ids = vec![
            "vec:000000000001".to_string(),
            "vec:000000000042".to_string(),
            "vec:000000000100".to_string(),
        ];

        let ids = extract_numeric_ids(&doc_ids, "vec:");
        assert_eq!(ids, vec![1, 42, 100]);
    }

    #[test]
    fn test_extract_numeric_ids_empty() {
        let doc_ids: Vec<String> = vec![];
        let ids = extract_numeric_ids(&doc_ids, "vec:");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_extract_numeric_ids_with_zero() {
        let doc_ids = vec![
            "vec:000000000000".to_string(),
            "vec:000000000001".to_string(),
        ];

        let ids = extract_numeric_ids(&doc_ids, "vec:");
        assert_eq!(ids, vec![0, 1]);
    }
}
