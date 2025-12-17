//! Glide-based control plane operations
//!
//! This module provides control plane operations using glide-core:
//! - Cluster discovery (CLUSTER NODES)
//! - INFO commands
//! - FT.* commands (CREATE, DROP, INFO, SEARCH)
//!
//! The data plane (benchmark traffic) still uses raw TCP connections
//! for maximum performance.

#[cfg(feature = "glide-control-plane")]
mod client;

#[cfg(feature = "glide-control-plane")]
pub use client::GlideControlPlane;
