//! FT.INFO response parsing
//!
//! Parses FT.INFO responses from different engine types:
//! - EC (ElastiCache Valkey) - uses flat key-value pairs
//! - MemoryDB - uses nested RESP3 structures

use std::collections::HashMap;

use crate::utils::RespValue;

/// Engine type for determining response format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineType {
    /// Unknown engine type
    Unknown,
    /// Open Source Valkey
    OssValkey,
    /// ElastiCache Valkey (provisioned)
    ElasticacheValkey,
    /// ElastiCache Serverless
    ElasticacheServerless,
    /// Amazon MemoryDB
    MemoryDb,
}

impl EngineType {
    /// Detect engine type from server info
    pub fn detect(info_response: &str) -> Self {
        // Check for MemoryDB specific indicators
        if info_response.contains("memorydb") || info_response.contains("MemoryDB") {
            return EngineType::MemoryDb;
        }

        // Check for ElastiCache indicators
        if info_response.contains("elasticache") || info_response.contains("ElastiCache") {
            if info_response.contains("serverless") {
                return EngineType::ElasticacheServerless;
            }
            return EngineType::ElasticacheValkey;
        }

        // Check for valkey-search module (EC Valkey)
        if info_response.contains("valkey-search") || info_response.contains("search_") {
            return EngineType::ElasticacheValkey;
        }

        EngineType::OssValkey
    }
}

/// Helper to get string from RespValue
fn resp_to_string(value: &RespValue) -> Option<String> {
    match value {
        RespValue::SimpleString(s) => Some(s.clone()),
        RespValue::BulkString(bytes) => String::from_utf8(bytes.clone()).ok(),
        _ => None,
    }
}

/// Convert FT.INFO RESP response to key:value lines (EC format)
///
/// EC format uses alternating key-value pairs in arrays:
/// ```text
/// 1) "index_name"
/// 2) "my-index"
/// 3) "num_docs"
/// 4) (integer) 1000
/// ```
pub fn convert_ftinfo_to_lines(reply: &RespValue, prefix: Option<&str>) -> String {
    let mut lines = String::new();
    convert_ftinfo_recursive(reply, prefix, &mut lines);
    lines
}

fn convert_ftinfo_recursive(reply: &RespValue, prefix: Option<&str>, lines: &mut String) {
    match reply {
        RespValue::SimpleString(s) => {
            if let Some(p) = prefix {
                lines.push_str(&format!("{}:{}\n", p, s));
            }
        }
        RespValue::BulkString(bytes) => {
            if let Some(p) = prefix {
                if let Ok(s) = std::str::from_utf8(bytes) {
                    lines.push_str(&format!("{}:{}\n", p, s));
                }
            }
        }
        RespValue::Integer(i) => {
            if let Some(p) = prefix {
                lines.push_str(&format!("{}:{}\n", p, i));
            }
        }
        RespValue::Array(elements) => {
            // Process as key-value pairs
            let mut i = 0;
            while i < elements.len() {
                let element = &elements[i];

                // Check if this element is a nested array
                if let RespValue::Array(_) = element {
                    convert_ftinfo_recursive(element, prefix, lines);
                    i += 1;
                    continue;
                }

                // Last element without a value
                if i == elements.len() - 1 {
                    if let Some(s) = resp_to_string(element) {
                        if let Some(p) = prefix {
                            lines.push_str(&format!("{}:{}\n", p, s));
                        }
                    } else if let RespValue::Integer(n) = element {
                        if let Some(p) = prefix {
                            lines.push_str(&format!("{}:{}\n", p, n));
                        }
                    }
                    break;
                }

                // Get key name
                let key_name = match resp_to_string(element) {
                    Some(s) => s,
                    None => {
                        i += 1;
                        continue;
                    }
                };

                // Build full key with prefix
                let full_key = match prefix {
                    Some(p) => format!("{}.{}", p, key_name),
                    None => key_name,
                };

                i += 1;
                if i >= elements.len() {
                    break;
                }

                let value = &elements[i];
                if let Some(s) = resp_to_string(value) {
                    lines.push_str(&format!("{}:{}\n", full_key, s));
                } else if let RespValue::Integer(n) = value {
                    lines.push_str(&format!("{}:{}\n", full_key, n));
                } else if let RespValue::Array(_) = value {
                    // Nested array - recurse with new prefix
                    convert_ftinfo_recursive(value, Some(&full_key), lines);
                }

                i += 1;
            }
        }
        _ => {}
    }
}

/// Convert FT.INFO RESP response to key:value lines (MemoryDB format)
///
/// MemoryDB format uses RESP3 maps with nested structures:
/// ```text
/// 1) index_name
/// 2) "gist-960-1M-960-100"
/// 3) fields
/// 4) 1) 1) identifier
///       2) vector_field
///       ...
/// ```
pub fn convert_memdb_ftinfo_to_lines(reply: &RespValue, prefix: Option<&str>) -> String {
    let mut lines = String::new();
    convert_memdb_recursive(reply, prefix, &mut lines);
    lines
}

fn convert_memdb_recursive(reply: &RespValue, prefix: Option<&str>, lines: &mut String) {
    match reply {
        RespValue::SimpleString(s) => {
            if let Some(p) = prefix {
                lines.push_str(&format!("{}:{}\n", p, s));
            }
        }
        RespValue::BulkString(bytes) => {
            if let Some(p) = prefix {
                if let Ok(s) = std::str::from_utf8(bytes) {
                    lines.push_str(&format!("{}:{}\n", p, s));
                }
            }
        }
        RespValue::Integer(i) => {
            if let Some(p) = prefix {
                lines.push_str(&format!("{}:{}\n", p, i));
            }
        }
        RespValue::Array(elements) => {
            // Process as key-value pairs (step by 2)
            let mut i = 0;
            while i + 1 < elements.len() {
                let key_elem = &elements[i];
                let val_elem = &elements[i + 1];

                // Key should be a string
                let key_name = match resp_to_string(key_elem) {
                    Some(s) => s,
                    None => {
                        i += 2;
                        continue;
                    }
                };

                // Build full key with prefix
                let full_key = match prefix {
                    Some(p) => format!("{}.{}", p, key_name),
                    None => key_name,
                };

                if let Some(s) = resp_to_string(val_elem) {
                    lines.push_str(&format!("{}:{}\n", full_key, s));
                } else if let RespValue::Integer(n) = val_elem {
                    lines.push_str(&format!("{}:{}\n", full_key, n));
                } else if let RespValue::Array(sub_elements) = val_elem {
                    if sub_elements.is_empty() {
                        i += 2;
                        continue;
                    }

                    // Check first element to determine structure
                    let first = &sub_elements[0];
                    match first {
                        // If first element is an array, it's a list of objects (like fields)
                        RespValue::Array(_) => {
                            for sub in sub_elements {
                                convert_memdb_recursive(sub, Some(&full_key), lines);
                            }
                        }
                        // If it's a string and we have even elements, it's key-value pairs
                        RespValue::SimpleString(_) | RespValue::BulkString(_)
                            if sub_elements.len() % 2 == 0 =>
                        {
                            convert_memdb_recursive(val_elem, Some(&full_key), lines);
                        }
                        // Otherwise it's a simple list - take first value
                        _ => {
                            if let Some(s) = resp_to_string(first) {
                                lines.push_str(&format!("{}:{}\n", full_key, s));
                            } else if let RespValue::Integer(n) = first {
                                lines.push_str(&format!("{}:{}\n", full_key, n));
                            }
                        }
                    }
                }

                i += 2;
            }
        }
        _ => {}
    }
}

/// Parse FT.INFO lines into a HashMap
pub fn parse_ftinfo_lines(lines: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for line in lines.lines() {
        if let Some((key, value)) = line.split_once(':') {
            map.insert(key.to_string(), value.to_string());
        }
    }
    map
}

/// Extract a specific field value from parsed FT.INFO
pub fn get_ftinfo_field<T: std::str::FromStr>(
    info: &HashMap<String, String>,
    field: &str,
) -> Option<T> {
    info.get(field)?.parse().ok()
}

/// Index status from FT.INFO
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexStatus {
    Available,
    Backfilling,
    Queued,
    Unknown(String),
}

impl IndexStatus {
    pub fn from_str(s: &str) -> Self {
        let upper = s.to_uppercase();
        match upper.as_str() {
            "AVAILABLE" | "READY" => IndexStatus::Available,
            "BACKFILLING" | "BACKFILL_IN_PROGRESS" => IndexStatus::Backfilling,
            s if s.contains("QUEUED") => IndexStatus::Queued,
            _ => IndexStatus::Unknown(s.to_string()),
        }
    }

    pub fn is_in_progress(&self) -> bool {
        matches!(self, IndexStatus::Backfilling | IndexStatus::Queued)
    }
}

/// Parsed FT.INFO result
#[derive(Debug, Clone)]
pub struct FtInfoResult {
    /// Index name
    pub index_name: Option<String>,
    /// Number of documents
    pub num_docs: i64,
    /// Number of indexed vectors (MemoryDB)
    pub num_indexed_vectors: i64,
    /// Index status
    pub status: IndexStatus,
    /// Index degradation percentage (MemoryDB)
    pub degradation_percentage: i32,
    /// Backfill in progress (EC)
    pub backfill_in_progress: bool,
    /// Backfill complete percent (EC)
    pub backfill_complete_percent: f64,
    /// Space usage bytes
    pub space_usage: i64,
    /// Vector space usage bytes
    pub vector_space_usage: i64,
    /// Current lag (MemoryDB)
    pub current_lag: i64,
    /// Raw parsed lines
    pub raw: HashMap<String, String>,
}

impl FtInfoResult {
    /// Parse from EC format response
    pub fn from_ec_response(reply: &RespValue) -> Self {
        let lines = convert_ftinfo_to_lines(reply, None);
        let raw = parse_ftinfo_lines(&lines);

        let backfill_in_progress: i32 = get_ftinfo_field(&raw, "backfill_in_progress").unwrap_or(0);
        let backfill_complete_percent: f64 =
            get_ftinfo_field(&raw, "backfill_complete_percent").unwrap_or(1.0);

        Self {
            index_name: raw.get("index_name").cloned(),
            num_docs: get_ftinfo_field(&raw, "num_docs").unwrap_or(0),
            num_indexed_vectors: get_ftinfo_field(&raw, "num_indexed_vectors").unwrap_or(0),
            status: raw
                .get("state")
                .map(|s| IndexStatus::from_str(s))
                .unwrap_or(IndexStatus::Available),
            degradation_percentage: 0,
            backfill_in_progress: backfill_in_progress != 0,
            backfill_complete_percent,
            space_usage: get_ftinfo_field(&raw, "space_usage").unwrap_or(0),
            vector_space_usage: get_ftinfo_field(&raw, "vector_space_usage").unwrap_or(0),
            current_lag: 0,
            raw,
        }
    }

    /// Parse from MemoryDB format response
    pub fn from_memdb_response(reply: &RespValue) -> Self {
        let lines = convert_memdb_ftinfo_to_lines(reply, None);
        let raw = parse_ftinfo_lines(&lines);

        Self {
            index_name: raw.get("index_name").cloned(),
            num_docs: get_ftinfo_field(&raw, "num_docs").unwrap_or(0),
            num_indexed_vectors: get_ftinfo_field(&raw, "num_indexed_vectors").unwrap_or(0),
            status: raw
                .get("index_status")
                .map(|s| IndexStatus::from_str(s))
                .unwrap_or(IndexStatus::Available),
            degradation_percentage: get_ftinfo_field(&raw, "index_degradation_percentage")
                .unwrap_or(0),
            backfill_in_progress: false,
            backfill_complete_percent: 100.0,
            space_usage: get_ftinfo_field(&raw, "space_usage").unwrap_or(0),
            vector_space_usage: get_ftinfo_field(&raw, "vector_space_usage").unwrap_or(0),
            current_lag: get_ftinfo_field(&raw, "current_lag").unwrap_or(0),
            raw,
        }
    }

    /// Parse from response based on engine type
    pub fn from_response(reply: &RespValue, engine_type: EngineType) -> Self {
        match engine_type {
            EngineType::MemoryDb => Self::from_memdb_response(reply),
            _ => Self::from_ec_response(reply),
        }
    }

    /// Check if index is ready (not backfilling/degraded)
    pub fn is_ready(&self) -> bool {
        self.status == IndexStatus::Available
            && self.degradation_percentage == 0
            && !self.backfill_in_progress
    }

    /// Get progress percentage (0-100)
    pub fn progress_percent(&self) -> i32 {
        if self.is_ready() {
            return 100;
        }

        if self.backfill_in_progress {
            return (self.backfill_complete_percent * 100.0) as i32;
        }

        if self.degradation_percentage > 0 {
            return 100 - self.degradation_percentage;
        }

        if self.status == IndexStatus::Backfilling {
            return 0;
        }

        100
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_ftinfo_simple() {
        let reply = RespValue::Array(vec![
            RespValue::BulkString(b"index_name".to_vec()),
            RespValue::BulkString(b"my-index".to_vec()),
            RespValue::BulkString(b"num_docs".to_vec()),
            RespValue::Integer(1000),
        ]);

        let lines = convert_ftinfo_to_lines(&reply, None);
        assert!(lines.contains("index_name:my-index"));
        assert!(lines.contains("num_docs:1000"));
    }

    #[test]
    fn test_convert_ftinfo_nested() {
        let reply = RespValue::Array(vec![
            RespValue::BulkString(b"attributes".to_vec()),
            RespValue::Array(vec![
                RespValue::BulkString(b"dim".to_vec()),
                RespValue::Integer(128),
            ]),
        ]);

        let lines = convert_ftinfo_to_lines(&reply, None);
        assert!(lines.contains("attributes.dim:128"));
    }

    #[test]
    fn test_index_status() {
        assert_eq!(IndexStatus::from_str("AVAILABLE"), IndexStatus::Available);
        assert_eq!(IndexStatus::from_str("BACKFILLING"), IndexStatus::Backfilling);
        assert!(IndexStatus::from_str("QUEUED_FOR_BACKFILL").is_in_progress());
    }

    #[test]
    fn test_engine_detection() {
        assert_eq!(
            EngineType::detect("# Server\r\nvalkey_version:7.2.0\r\n"),
            EngineType::OssValkey
        );
        assert_eq!(
            EngineType::detect("# Modules\r\nmodule:name=valkey-search\r\n"),
            EngineType::ElasticacheValkey
        );
        assert_eq!(
            EngineType::detect("# Server\r\nmemorydb_version:7.1.0\r\n"),
            EngineType::MemoryDb
        );
    }
}
