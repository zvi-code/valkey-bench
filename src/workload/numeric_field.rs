//! Numeric field configuration for benchmarks
//!
//! This module provides configurable numeric field generation with:
//! - Multiple value types (Int, Float, Unix timestamp, ISO date)
//! - Multiple distributions (Uniform, Zipfian, Normal, Sequential)
//!
//! ## CLI Syntax
//!
//! ```text
//! --numeric-field "field_name:type:distribution:param1:param2..."
//!
//! Types:
//!   int          - Integer value
//!   float        - Floating point (default 2 decimal places)
//!   float3       - Floating point with 3 decimal places
//!   unix         - Unix timestamp (seconds since epoch)
//!   iso          - ISO 8601 datetime (2024-01-15T10:30:00Z)
//!   date         - Date only (2024-01-15)
//!
//! Distributions:
//!   uniform:min:max     - Uniform random in [min, max)
//!   zipfian:skew:min:max - Zipfian (power-law) distribution
//!   normal:mean:stddev  - Normal/Gaussian distribution
//!   seq:start:step      - Sequential values
//!   const:value         - Constant value
//!   key                 - Based on key number (deterministic)
//! ```
//!
//! ## Examples
//!
//! ```text
//! --numeric-field "timestamp:unix:uniform:1609459200:1704067200"
//! --numeric-field "price:float2:normal:100.0:15.0"
//! --numeric-field "score:int:zipfian:0.99:0:1000"
//! --numeric-field "version:int:seq:1:1"
//! ```

use std::fmt::Write;

/// Numeric field configuration
#[derive(Debug, Clone)]
pub struct NumericFieldConfig {
    /// Field name (user-defined)
    pub name: String,
    /// Value type (affects formatting)
    pub value_type: NumericValueType,
    /// Distribution for value generation
    pub distribution: NumericDistribution,
}

/// Numeric value type (affects formatting)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumericValueType {
    /// Integer (formatted as decimal)
    Int,
    /// Floating point with specified decimal places
    Float { precision: u8 },
    /// Unix timestamp (seconds since epoch)
    UnixTimestamp,
    /// ISO 8601 datetime string (2024-01-15T10:30:00Z)
    IsoDateTime,
    /// Date only (2024-01-15)
    DateOnly,
}

/// Distribution for numeric value generation
#[derive(Debug, Clone)]
pub enum NumericDistribution {
    /// Uniform random distribution in [min, max)
    Uniform { min: f64, max: f64 },
    /// Zipfian (power-law) distribution - models "hot" values
    Zipfian { skew: f64, min: f64, max: f64 },
    /// Normal/Gaussian distribution
    Normal { mean: f64, stddev: f64 },
    /// Sequential values (increments with each generation)
    Sequential { start: f64, step: f64 },
    /// Constant value
    Constant { value: f64 },
    /// Key-based (deterministic based on key_num, scaled to range)
    KeyBased { min: f64, max: f64 },
}

impl NumericFieldConfig {
    /// Create a new numeric field config with uniform distribution
    pub fn new_uniform(name: &str, value_type: NumericValueType, min: f64, max: f64) -> Self {
        Self {
            name: name.to_string(),
            value_type,
            distribution: NumericDistribution::Uniform { min, max },
        }
    }

    /// Create a simple integer field with key-based values
    pub fn new_key_based(name: &str, min: f64, max: f64) -> Self {
        Self {
            name: name.to_string(),
            value_type: NumericValueType::Int,
            distribution: NumericDistribution::KeyBased { min, max },
        }
    }

    /// Parse from CLI string format
    ///
    /// Format: "name:type:distribution:params..."
    pub fn parse(input: &str) -> Result<Self, String> {
        let parts: Vec<&str> = input.split(':').collect();
        if parts.len() < 3 {
            return Err(format!(
                "Invalid numeric field format: '{}'. Expected 'name:type:distribution:params...'",
                input
            ));
        }

        let name = parts[0].to_string();
        let value_type = Self::parse_value_type(parts[1])?;
        let distribution = Self::parse_distribution(&parts[2..])?;

        Ok(Self {
            name,
            value_type,
            distribution,
        })
    }

    /// Parse value type from string
    fn parse_value_type(s: &str) -> Result<NumericValueType, String> {
        match s.to_lowercase().as_str() {
            "int" => Ok(NumericValueType::Int),
            "float" => Ok(NumericValueType::Float { precision: 2 }),
            "unix" => Ok(NumericValueType::UnixTimestamp),
            "iso" => Ok(NumericValueType::IsoDateTime),
            "date" => Ok(NumericValueType::DateOnly),
            s if s.starts_with("float") => {
                let precision: u8 = s[5..]
                    .parse()
                    .map_err(|_| format!("Invalid float precision in '{}'", s))?;
                Ok(NumericValueType::Float { precision })
            }
            _ => Err(format!("Unknown numeric type '{}'. Expected: int, float, floatN, unix, iso, date", s)),
        }
    }

    /// Parse distribution from parts (starting from distribution name)
    fn parse_distribution(parts: &[&str]) -> Result<NumericDistribution, String> {
        if parts.is_empty() {
            return Err("Missing distribution".to_string());
        }

        match parts[0].to_lowercase().as_str() {
            "uniform" => {
                if parts.len() < 3 {
                    return Err("uniform requires min:max parameters".to_string());
                }
                let min: f64 = parts[1].parse().map_err(|_| "Invalid min value")?;
                let max: f64 = parts[2].parse().map_err(|_| "Invalid max value")?;
                Ok(NumericDistribution::Uniform { min, max })
            }
            "zipfian" => {
                if parts.len() < 4 {
                    return Err("zipfian requires skew:min:max parameters".to_string());
                }
                let skew: f64 = parts[1].parse().map_err(|_| "Invalid skew value")?;
                let min: f64 = parts[2].parse().map_err(|_| "Invalid min value")?;
                let max: f64 = parts[3].parse().map_err(|_| "Invalid max value")?;
                Ok(NumericDistribution::Zipfian { skew, min, max })
            }
            "normal" => {
                if parts.len() < 3 {
                    return Err("normal requires mean:stddev parameters".to_string());
                }
                let mean: f64 = parts[1].parse().map_err(|_| "Invalid mean value")?;
                let stddev: f64 = parts[2].parse().map_err(|_| "Invalid stddev value")?;
                Ok(NumericDistribution::Normal { mean, stddev })
            }
            "seq" => {
                if parts.len() < 3 {
                    return Err("seq requires start:step parameters".to_string());
                }
                let start: f64 = parts[1].parse().map_err(|_| "Invalid start value")?;
                let step: f64 = parts[2].parse().map_err(|_| "Invalid step value")?;
                Ok(NumericDistribution::Sequential { start, step })
            }
            "const" => {
                if parts.len() < 2 {
                    return Err("const requires value parameter".to_string());
                }
                let value: f64 = parts[1].parse().map_err(|_| "Invalid const value")?;
                Ok(NumericDistribution::Constant { value })
            }
            "key" => {
                // Optional min:max for scaling, defaults to 0:key_num
                let (min, max) = if parts.len() >= 3 {
                    let min: f64 = parts[1].parse().map_err(|_| "Invalid min value")?;
                    let max: f64 = parts[2].parse().map_err(|_| "Invalid max value")?;
                    (min, max)
                } else {
                    (0.0, f64::MAX)
                };
                Ok(NumericDistribution::KeyBased { min, max })
            }
            _ => Err(format!(
                "Unknown distribution '{}'. Expected: uniform, zipfian, normal, seq, const, key",
                parts[0]
            )),
        }
    }

    /// Maximum byte length needed for this field's formatted value
    pub fn max_byte_len(&self) -> usize {
        match self.value_type {
            NumericValueType::Int => 20, // Max i64 length
            NumericValueType::Float { precision } => 20 + precision as usize, // Digits + decimal + precision
            NumericValueType::UnixTimestamp => 12, // 10-digit timestamp + room for growth
            NumericValueType::IsoDateTime => 24, // 2024-01-15T10:30:00Z
            NumericValueType::DateOnly => 10, // 2024-01-15
        }
    }

    /// Generate a value using the configured distribution
    ///
    /// # Arguments
    /// * `key_num` - Key number (used for deterministic generation)
    /// * `seq_counter` - Sequential counter (used for Sequential distribution)
    pub fn generate_value(&self, key_num: u64, seq_counter: u64) -> f64 {
        match &self.distribution {
            NumericDistribution::Uniform { min, max } => {
                // Use key_num as seed for reproducibility
                let mut rng = fastrand::Rng::with_seed(key_num);
                min + rng.f64() * (max - min)
            }
            NumericDistribution::Zipfian { skew, min, max } => {
                // Zipfian distribution using rejection sampling
                let mut rng = fastrand::Rng::with_seed(key_num);
                let range = max - min;
                // Simplified zipfian: use power-law transformation
                let u = rng.f64();
                let zipf_val = ((1.0 - u).powf(1.0 - skew) - 1.0) / ((1.0 - skew) - 1.0);
                min + (zipf_val.abs() % 1.0) * range
            }
            NumericDistribution::Normal { mean, stddev } => {
                // Box-Muller transform for normal distribution
                let mut rng = fastrand::Rng::with_seed(key_num);
                let u1 = rng.f64();
                let u2 = rng.f64();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                mean + z * stddev
            }
            NumericDistribution::Sequential { start, step } => {
                start + (seq_counter as f64) * step
            }
            NumericDistribution::Constant { value } => *value,
            NumericDistribution::KeyBased { min, max } => {
                if *max == f64::MAX {
                    key_num as f64
                } else {
                    let range = max - min;
                    min + (key_num as f64 % range)
                }
            }
        }
    }

    /// Format value to buffer, returns bytes written
    pub fn format_value(&self, value: f64, buf: &mut [u8]) -> usize {
        let mut tmp = String::with_capacity(32);

        match self.value_type {
            NumericValueType::Int => {
                write!(&mut tmp, "{}", value as i64).unwrap();
            }
            NumericValueType::Float { precision } => {
                write!(&mut tmp, "{:.prec$}", value, prec = precision as usize).unwrap();
            }
            NumericValueType::UnixTimestamp => {
                write!(&mut tmp, "{}", value as i64).unwrap();
            }
            NumericValueType::IsoDateTime => {
                // Convert Unix timestamp to ISO 8601
                let secs = value as i64;
                let days_since_epoch = secs / 86400;
                let time_of_day = secs % 86400;
                let hours = time_of_day / 3600;
                let minutes = (time_of_day % 3600) / 60;
                let seconds = time_of_day % 60;

                // Calculate year, month, day from days since epoch (1970-01-01)
                let (year, month, day) = days_to_ymd(days_since_epoch);
                write!(
                    &mut tmp,
                    "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
                    year, month, day, hours, minutes, seconds
                )
                .unwrap();
            }
            NumericValueType::DateOnly => {
                let secs = value as i64;
                let days_since_epoch = secs / 86400;
                let (year, month, day) = days_to_ymd(days_since_epoch);
                write!(&mut tmp, "{:04}-{:02}-{:02}", year, month, day).unwrap();
            }
        }

        let bytes = tmp.as_bytes();
        let len = bytes.len().min(buf.len());
        buf[..len].copy_from_slice(&bytes[..len]);
        len
    }

    /// Fill buffer with formatted value, pad with zeros if needed
    pub fn fill_buffer(&self, key_num: u64, seq_counter: u64, buf: &mut [u8]) {
        let value = self.generate_value(key_num, seq_counter);
        let written = self.format_value(value, buf);
        // Pad remainder with zeros
        buf[written..].fill(b'0');
    }
}

/// Convert days since Unix epoch to (year, month, day)
fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    // Simplified algorithm for converting days to date
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year as i32, m, d)
}

/// Collection of numeric field configurations
#[derive(Debug, Clone, Default)]
pub struct NumericFieldSet {
    fields: Vec<NumericFieldConfig>,
}

impl NumericFieldSet {
    pub fn new() -> Self {
        Self { fields: Vec::new() }
    }

    pub fn add(&mut self, config: NumericFieldConfig) {
        self.fields.push(config);
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    pub fn len(&self) -> usize {
        self.fields.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &NumericFieldConfig> {
        self.fields.iter()
    }

    pub fn get(&self, idx: usize) -> Option<&NumericFieldConfig> {
        self.fields.get(idx)
    }

    /// Parse from CLI string with multiple fields
    /// Format: "field1:type:dist:params,field2:type:dist:params"
    pub fn parse(input: &str) -> Result<Self, String> {
        let mut set = Self::new();
        for field_str in input.split(',') {
            let field_str = field_str.trim();
            if !field_str.is_empty() {
                set.add(NumericFieldConfig::parse(field_str)?);
            }
        }
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_int_uniform() {
        let config = NumericFieldConfig::parse("score:int:uniform:0:100").unwrap();
        assert_eq!(config.name, "score");
        assert_eq!(config.value_type, NumericValueType::Int);
        assert!(matches!(
            config.distribution,
            NumericDistribution::Uniform { min: 0.0, max: 100.0 }
        ));
    }

    #[test]
    fn test_parse_float_normal() {
        let config = NumericFieldConfig::parse("price:float2:normal:50.0:10.0").unwrap();
        assert_eq!(config.name, "price");
        assert!(matches!(config.value_type, NumericValueType::Float { precision: 2 }));
        assert!(matches!(
            config.distribution,
            NumericDistribution::Normal { mean: 50.0, stddev: 10.0 }
        ));
    }

    #[test]
    fn test_parse_unix_timestamp() {
        let config = NumericFieldConfig::parse("created:unix:uniform:1609459200:1704067200").unwrap();
        assert_eq!(config.value_type, NumericValueType::UnixTimestamp);
    }

    #[test]
    fn test_parse_iso_datetime() {
        let config = NumericFieldConfig::parse("updated:iso:key:1609459200:1704067200").unwrap();
        assert_eq!(config.value_type, NumericValueType::IsoDateTime);
    }

    #[test]
    fn test_parse_sequential() {
        let config = NumericFieldConfig::parse("version:int:seq:1:1").unwrap();
        assert!(matches!(
            config.distribution,
            NumericDistribution::Sequential { start: 1.0, step: 1.0 }
        ));
    }

    #[test]
    fn test_generate_uniform() {
        let config = NumericFieldConfig::new_uniform("test", NumericValueType::Int, 0.0, 100.0);
        let v1 = config.generate_value(42, 0);
        let v2 = config.generate_value(42, 0);
        assert_eq!(v1, v2); // Same seed = same value
        assert!(v1 >= 0.0 && v1 < 100.0);
    }

    #[test]
    fn test_generate_sequential() {
        let config = NumericFieldConfig::parse("v:int:seq:10:5").unwrap();
        assert_eq!(config.generate_value(0, 0), 10.0);
        assert_eq!(config.generate_value(0, 1), 15.0);
        assert_eq!(config.generate_value(0, 2), 20.0);
    }

    #[test]
    fn test_format_int() {
        let config = NumericFieldConfig::new_uniform("test", NumericValueType::Int, 0.0, 100.0);
        let mut buf = [0u8; 20];
        let len = config.format_value(42.7, &mut buf);
        assert_eq!(&buf[..len], b"42");
    }

    #[test]
    fn test_format_float() {
        let config = NumericFieldConfig::parse("test:float3:const:3.14159").unwrap();
        let mut buf = [0u8; 20];
        let len = config.format_value(3.14159, &mut buf);
        assert_eq!(&buf[..len], b"3.142");
    }

    #[test]
    fn test_format_iso_datetime() {
        let config = NumericFieldConfig::parse("test:iso:const:1704067200").unwrap();
        let mut buf = [0u8; 30];
        let len = config.format_value(1704067200.0, &mut buf);
        // 2024-01-01T00:00:00Z
        assert!(std::str::from_utf8(&buf[..len]).unwrap().starts_with("2024-01-01"));
    }

    #[test]
    fn test_parse_multiple_fields() {
        let set = NumericFieldSet::parse("score:int:uniform:0:100,price:float2:const:99.99").unwrap();
        assert_eq!(set.len(), 2);
        assert_eq!(set.get(0).unwrap().name, "score");
        assert_eq!(set.get(1).unwrap().name, "price");
    }
}
