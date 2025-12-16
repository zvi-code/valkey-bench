//! Dataset header structure (C-compatible)
//!
//! This module defines the binary header format for vector datasets.
//! The structure matches the C implementation exactly for compatibility.

/// Dataset magic number to identify valid files
pub const DATASET_MAGIC: u32 = 0xDECDB001;

/// Header size in bytes (always 4096 for alignment)
pub const HEADER_SIZE: usize = 4096;

/// Data type identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DataType {
    Float32 = 0,
    Float16 = 1,
}

impl DataType {
    /// Get element size in bytes
    pub fn element_size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float16 => 2,
        }
    }

    /// Parse from raw byte value
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => DataType::Float16,
            _ => DataType::Float32, // Default to float32
        }
    }
}

/// Distance metric identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DistanceMetricId {
    L2 = 0,
    InnerProduct = 1,
    Cosine = 2,
}

impl DistanceMetricId {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            DistanceMetricId::L2 => "L2",
            DistanceMetricId::InnerProduct => "IP",
            DistanceMetricId::Cosine => "COSINE",
        }
    }

    /// Parse from raw byte value
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => DistanceMetricId::InnerProduct,
            2 => DistanceMetricId::Cosine,
            _ => DistanceMetricId::L2,
        }
    }
}

/// Binary dataset header - matches Python prepare_binary.py format
///
/// This structure is read directly from the file header.
/// The packed representation ensures byte-for-byte compatibility.
///
/// Python struct format: '<II256sBBxxIQQIxxxxQQQ{padding}x'
///   - I: magic (4 bytes)
///   - I: version (4 bytes)
///   - 256s: name (256 bytes)
///   - B: distance_metric (1 byte)
///   - B: dtype (1 byte)
///   - xx: padding (2 bytes)
///   - I: dim (4 bytes)
///   - Q: num_vectors (8 bytes)
///   - Q: num_queries (8 bytes)
///   - I: num_neighbors (4 bytes)
///   - xxxx: padding (4 bytes)
///   - Q: vectors_offset (8 bytes)
///   - Q: queries_offset (8 bytes)
///   - Q: ground_truth_offset (8 bytes)
///   - Total base: 320 bytes, padding: 3776 bytes = 4096 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct DatasetHeader {
    /// Magic number (0xDECDB001)
    pub magic: u32,
    /// Format version
    pub version: u32,
    /// Dataset name (null-terminated, max 255 chars)
    pub dataset_name: [u8; 256],
    /// Distance metric (0=L2, 1=COSINE, 2=IP)
    pub distance_metric: u8,
    /// Data type (0=FLOAT32, 1=FLOAT16)
    pub dtype: u8,
    /// Padding for alignment
    pub _padding1: [u8; 2],
    /// Vector dimension
    pub dim: u32,
    /// Total number of database vectors
    pub num_vectors: u64,
    /// Number of query vectors
    pub num_queries: u64,
    /// Number of ground truth neighbors per query
    pub num_neighbors: u32,
    /// Padding for alignment
    pub _padding2: [u8; 4],
    /// File offset to vector data
    pub vectors_offset: u64,
    /// File offset to query vectors
    pub queries_offset: u64,
    /// File offset to ground truth neighbors
    pub ground_truth_offset: u64,
    /// Reserved for future use (pads header to 4096 bytes)
    pub reserved: [u8; 3776],
}

// Ensure the header is exactly 4096 bytes
const _: () = assert!(std::mem::size_of::<DatasetHeader>() == HEADER_SIZE);

impl DatasetHeader {
    /// Get dataset name as string
    pub fn name(&self) -> &str {
        let end = self
            .dataset_name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.dataset_name.len());
        std::str::from_utf8(&self.dataset_name[..end]).unwrap_or("unknown")
    }

    /// Get data type enum
    pub fn data_type(&self) -> DataType {
        DataType::from_u8(self.dtype)
    }

    /// Get distance metric enum
    pub fn distance_metric_type(&self) -> DistanceMetricId {
        DistanceMetricId::from_u8(self.distance_metric)
    }

    /// Get distance metric name
    pub fn distance_metric_str(&self) -> &'static str {
        self.distance_metric_type().as_str()
    }

    /// Get vector byte length based on dimension and data type
    pub fn vec_byte_len(&self) -> usize {
        let elem_size = self.data_type().element_size();
        self.dim as usize * elem_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<DatasetHeader>(), HEADER_SIZE);
    }

    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::Float32.element_size(), 4);
        assert_eq!(DataType::Float16.element_size(), 2);
    }

    #[test]
    fn test_distance_metric_str() {
        assert_eq!(DistanceMetricId::L2.as_str(), "L2");
        assert_eq!(DistanceMetricId::InnerProduct.as_str(), "IP");
        assert_eq!(DistanceMetricId::Cosine.as_str(), "COSINE");
    }

    #[test]
    fn test_vec_byte_len() {
        // Create a mock header
        let mut header = DatasetHeader {
            magic: DATASET_MAGIC,
            version: 1,
            dataset_name: [0u8; 256],
            distance_metric: 0,
            dtype: 0, // FLOAT32
            _padding1: [0; 2],
            dim: 128,
            num_vectors: 1000,
            num_queries: 100,
            num_neighbors: 10,
            _padding2: [0; 4],
            vectors_offset: HEADER_SIZE as u64,
            queries_offset: 0,
            ground_truth_offset: 0,
            reserved: [0u8; 3776],
        };

        // FLOAT32 with dim=128: 128 * 4 = 512
        assert_eq!(header.vec_byte_len(), 512);

        // FLOAT16 with dim=128: 128 * 2 = 256
        header.dtype = 1;
        assert_eq!(header.vec_byte_len(), 256);
    }
}
