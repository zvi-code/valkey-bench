//! Workload definitions and command templates

pub mod command_template;
pub mod search_ops;
pub mod template_factory;
pub mod workload_type;

pub use command_template::{CommandTemplate, TemplateArg};
pub use search_ops::{
    create_index, drop_index, extract_numeric_ids, get_index_info, parse_search_response,
    wait_for_indexing, IndexInfo,
};
pub use template_factory::create_template;
pub use workload_type::WorkloadType;
