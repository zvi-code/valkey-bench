//! Template factory for creating command templates for all workload types

use crate::config::SearchConfig;

use super::command_template::CommandTemplate;
use super::key_format::DEFAULT_KEY_WIDTH;
use super::workload_type::WorkloadType;

/// Create command template for given workload type
/// 
/// When `cluster_mode` is true, key-value commands will use cluster-tagged keys
/// (e.g., `key:{ABC}:000000000001`) to ensure proper routing without MOVED redirects.
pub fn create_template(
    workload: WorkloadType,
    key_prefix: &str,
    data_size: usize,
    search_config: Option<&SearchConfig>,
    cluster_mode: bool,
) -> CommandTemplate {
    let key_width = DEFAULT_KEY_WIDTH;

    // Helper to create key arg based on cluster mode
    let add_key = |template: CommandTemplate| -> CommandTemplate {
        if cluster_mode {
            template.arg_prefixed_key_with_cluster_tag(key_prefix, key_width)
        } else {
            template.arg_prefixed_key(key_prefix, key_width)
        }
    };

    match workload {
        // === Simple commands ===
        WorkloadType::Ping => CommandTemplate::new("PING").arg_str("PING"),

        // === Key-value commands ===
        WorkloadType::Set => add_key(CommandTemplate::new("SET").arg_str("SET"))
            .arg_literal(&vec![b'x'; data_size]),

        WorkloadType::Get => add_key(CommandTemplate::new("GET").arg_str("GET")),

        WorkloadType::Incr => add_key(CommandTemplate::new("INCR").arg_str("INCR")),

        // === List commands ===
        WorkloadType::Lpush => add_key(CommandTemplate::new("LPUSH").arg_str("LPUSH"))
            .arg_literal(&vec![b'x'; data_size]),

        WorkloadType::Rpush => add_key(CommandTemplate::new("RPUSH").arg_str("RPUSH"))
            .arg_literal(&vec![b'x'; data_size]),

        WorkloadType::Lpop => add_key(CommandTemplate::new("LPOP").arg_str("LPOP")),

        WorkloadType::Rpop => add_key(CommandTemplate::new("RPOP").arg_str("RPOP")),

        WorkloadType::Lrange100 => create_lrange_template(key_prefix, key_width, 100, cluster_mode),
        WorkloadType::Lrange300 => create_lrange_template(key_prefix, key_width, 300, cluster_mode),
        WorkloadType::Lrange500 => create_lrange_template(key_prefix, key_width, 500, cluster_mode),
        WorkloadType::Lrange600 => create_lrange_template(key_prefix, key_width, 600, cluster_mode),

        // === Set commands ===
        WorkloadType::Sadd => add_key(CommandTemplate::new("SADD").arg_str("SADD"))
            .arg_rand_int(key_width),

        WorkloadType::Spop => add_key(CommandTemplate::new("SPOP").arg_str("SPOP")),

        // === Hash commands ===
        WorkloadType::Hset => add_key(CommandTemplate::new("HSET").arg_str("HSET"))
            .arg_str("field")
            .arg_literal(&vec![b'x'; data_size]),

        // === Sorted set commands ===
        WorkloadType::Zadd => add_key(CommandTemplate::new("ZADD").arg_str("ZADD"))
            .arg_rand_int(key_width) // score
            .arg_str("member"),

        WorkloadType::Zpopmin => add_key(CommandTemplate::new("ZPOPMIN").arg_str("ZPOPMIN")),

        // === Multi-key commands ===
        WorkloadType::Mset => create_mset_template(key_prefix, key_width, data_size, 10, cluster_mode),

        // === Vector search commands (always use cluster tags) ===
        WorkloadType::VecLoad => {
            let sc = search_config.expect("VecLoad requires search config");
            create_vec_load_template(sc, key_width)
        }

        WorkloadType::VecQuery => {
            let sc = search_config.expect("VecQuery requires search config");
            create_vec_query_template(sc)
        }

        WorkloadType::VecDelete => {
            let sc = search_config.expect("VecDelete requires search config");
            // Use cluster tag format for consistency with VecLoad
            CommandTemplate::new("DEL")
                .arg_str("DEL")
                .arg_prefixed_key_with_cluster_tag(&sc.prefix, key_width)
        }

        WorkloadType::VecUpdate => {
            let sc = search_config.expect("VecUpdate requires search config");
            create_vec_load_template(sc, key_width)
        }

        WorkloadType::Custom => {
            // Custom commands should be handled separately
            CommandTemplate::new("CUSTOM").arg_str("PING")
        }
    }
}

/// Create LRANGE template with specified count
fn create_lrange_template(key_prefix: &str, key_width: usize, count: i32, cluster_mode: bool) -> CommandTemplate {
    let template = CommandTemplate::new(&format!("LRANGE_{}", count))
        .arg_str("LRANGE");
    
    let template = if cluster_mode {
        template.arg_prefixed_key_with_cluster_tag(key_prefix, key_width)
    } else {
        template.arg_prefixed_key(key_prefix, key_width)
    };
    
    template
        .arg_str("0")
        .arg_str(&(count - 1).to_string())
}

/// Create MSET template with multiple keys
fn create_mset_template(
    key_prefix: &str,
    key_width: usize,
    data_size: usize,
    num_keys: usize,
    cluster_mode: bool,
) -> CommandTemplate {
    let mut template = CommandTemplate::new("MSET").arg_str("MSET");

    for _ in 0..num_keys {
        template = if cluster_mode {
            template.arg_prefixed_key_with_cluster_tag(key_prefix, key_width)
        } else {
            template.arg_prefixed_key(key_prefix, key_width)
        };
        template = template.arg_literal(&vec![b'x'; data_size]);
    }

    template
}

/// Create HSET template for vector loading
/// Key format: prefix{tag}:vector_id (e.g., "zvec_:{ABC}:000000055083")
///
/// Fields added:
/// - vector_field: <vector data> (always)
/// - tag_field: <tag value> (if search_config.tag_field is set)
/// - numeric_field(s): <numeric value> (from search_config.numeric_fields)
fn create_vec_load_template(search_config: &SearchConfig, key_width: usize) -> CommandTemplate {
    // Key format: prefix + cluster_tag + ":" + vector_id
    // We use a compound key with cluster tag for proper shard distribution
    let mut template = CommandTemplate::new("HSET")
        .arg_str("HSET")
        .arg_prefixed_key_with_cluster_tag(&search_config.prefix, key_width)
        .arg_str(&search_config.vector_field)
        .arg_vector(search_config.vec_byte_len());

    // Add tag field if configured
    if let Some(ref tag_field) = search_config.tag_field {
        template = template
            .arg_str(tag_field)
            .arg_tag_placeholder(search_config.tag_max_len);
    }

    // Add numeric fields from the NumericFieldSet
    for (idx, field_config) in search_config.numeric_fields.iter().enumerate() {
        template = template
            .arg_str(&field_config.name)
            .arg_numeric_field(idx, field_config.max_byte_len());
    }

    template
}

/// Create FT.SEARCH template for vector queries
///
/// If tag_filter is set, the query changes from:
///   "*=>[KNN $K @embedding $BLOB]"
/// to:
///   "@tag_field:{filter}=>[KNN $K @embedding $BLOB]"
fn create_vec_query_template(search_config: &SearchConfig) -> CommandTemplate {
    let mut template = CommandTemplate::new("FT.SEARCH")
        .arg_str("FT.SEARCH")
        .arg_str(&search_config.index_name);

    // Build filter prefix based on tag_filter
    let filter_prefix = if let (Some(ref tag_field), Some(ref tag_filter)) =
        (&search_config.tag_field, &search_config.tag_filter)
    {
        format!("@{}:{{{}}}", tag_field, tag_filter)
    } else {
        "*".to_string()
    };

    // Build query string based on config
    let query = format!(
        "{}=>[KNN {} @{} $BLOB{}]",
        filter_prefix,
        search_config.k,
        search_config.vector_field,
        if let Some(ef) = search_config.ef_search {
            format!(" EF_RUNTIME {}", ef)
        } else {
            String::new()
        }
    );

    template = template
        .arg_str(&query)
        .arg_str("PARAMS")
        .arg_str("2") // 2 parameters
        .arg_str("BLOB")
        .arg_query_vector(search_config.vec_byte_len()) // Use query vector for FT.SEARCH
        .arg_str("DIALECT")
        .arg_str("2"); // DIALECT 2 required for KNN queries

    if search_config.nocontent {
        template = template.arg_str("NOCONTENT");
    }

    template
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DistanceMetric, VectorAlgorithm};
    use crate::workload::NumericFieldSet;

    #[test]
    fn test_create_ping_template() {
        let template = create_template(WorkloadType::Ping, "key:", 3, None, false);
        let buf = template.build(1);
        assert!(buf.placeholders[0].is_empty());
    }

    #[test]
    fn test_create_set_template() {
        let template = create_template(WorkloadType::Set, "key:", 100, None, false);
        let buf = template.build(1);
        assert_eq!(buf.placeholders[0].len(), 1); // Key placeholder
    }

    #[test]
    fn test_create_set_template_cluster_mode() {
        let template = create_template(WorkloadType::Set, "key:", 100, None, true);
        let buf = template.build(1);
        // In cluster mode: cluster_tag + key = 2 placeholders
        assert_eq!(buf.placeholders[0].len(), 2);
    }

    #[test]
    fn test_create_get_template() {
        let template = create_template(WorkloadType::Get, "key:", 3, None, false);
        let buf = template.build(1);
        assert_eq!(buf.placeholders[0].len(), 1); // Key placeholder
    }

    #[test]
    fn test_create_vec_load_template() {
        let search_config = SearchConfig {
            index_name: "idx".to_string(),
            vector_field: "embedding".to_string(),
            prefix: "vec:".to_string(),
            algorithm: VectorAlgorithm::Hnsw,
            distance_metric: DistanceMetric::L2,
            dim: 128,
            k: 10,
            ef_construction: None,
            hnsw_m: None,
            ef_search: None,
            nocontent: false,
            tag_field: None,
            tag_distributions: None,
            tag_filter: None,
            tag_max_len: 128,
            numeric_field: None,
            numeric_fields: NumericFieldSet::new(),
        };

        let template = create_template(WorkloadType::VecLoad, "key:", 3, Some(&search_config), false);
        let buf = template.build(1);

        // Should have cluster_tag, key, and vector placeholders
        // VecLoad uses arg_prefixed_key_with_cluster_tag which creates 2 placeholders (tag + key)
        // Plus 1 for the vector = 3 total
        assert_eq!(buf.placeholders[0].len(), 3);
    }
}
