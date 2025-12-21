//! Workload definitions and command templates

pub mod command_template;
pub mod key_format;
pub mod search_ops;
pub mod tag_distribution;
pub mod template_factory;
pub mod workload_type;

pub use command_template::{CommandTemplate, TemplateArg};
pub use key_format::{
    extract_numeric_ids_from_keys, KeyFormat, CLUSTER_TAG_INNER_LEN, CLUSTER_TAG_LEN,
    DEFAULT_KEY_WIDTH, TAG_KEY_SEPARATOR,
};
pub use search_ops::{
    create_index, drop_index, extract_numeric_ids, get_index_info, parse_search_response,
    wait_for_indexing, IndexInfo,
};
pub use tag_distribution::{TagDistribution, TagDistributionSet};
pub use template_factory::create_template;
pub use workload_type::WorkloadType;
