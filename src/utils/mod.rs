//! Utility modules

pub mod error;
pub mod resp;

pub use error::{
    BenchmarkError, ClusterError, ConnectionError, DatasetError, ProtocolError, Result,
};
pub use resp::{RespDecoder, RespEncoder, RespValue};
