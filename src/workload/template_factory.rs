//! Template factory for creating command templates for all workload types

use crate::config::SearchConfig;

use super::command_template::CommandTemplate;
use super::workload_type::WorkloadType;

/// Default key width for numeric keys
pub const DEFAULT_KEY_WIDTH: usize = 12;

/// Create command template for given workload type
pub fn create_template(
    workload: WorkloadType,
    key_prefix: &str,
    data_size: usize,
    search_config: Option<&SearchConfig>,
) -> CommandTemplate {
    let key_width = DEFAULT_KEY_WIDTH;

    match workload {
        // === Simple commands ===
        WorkloadType::Ping => CommandTemplate::new("PING").arg_str("PING"),

        // === Key-value commands ===
        WorkloadType::Set => CommandTemplate::new("SET")
            .arg_str("SET")
            .arg_prefixed_key(key_prefix, key_width)
            .arg_literal(&vec![b'x'; data_size]),

        WorkloadType::Get => CommandTemplate::new("GET")
            .arg_str("GET")
            .arg_prefixed_key(key_prefix, key_width),

        WorkloadType::Incr => CommandTemplate::new("INCR")
            .arg_str("INCR")
            .arg_prefixed_key(key_prefix, key_width),

        // === List commands ===
        WorkloadType::Lpush => CommandTemplate::new("LPUSH")
            .arg_str("LPUSH")
            .arg_prefixed_key(key_prefix, key_width)
            .arg_literal(&vec![b'x'; data_size]),

        WorkloadType::Rpush => CommandTemplate::new("RPUSH")
            .arg_str("RPUSH")
            .arg_prefixed_key(key_prefix, key_width)
            .arg_literal(&vec![b'x'; data_size]),

        WorkloadType::Lpop => CommandTemplate::new("LPOP")
            .arg_str("LPOP")
            .arg_prefixed_key(key_prefix, key_width),

        WorkloadType::Rpop => CommandTemplate::new("RPOP")
            .arg_str("RPOP")
            .arg_prefixed_key(key_prefix, key_width),

        WorkloadType::Lrange100 => create_lrange_template(key_prefix, key_width, 100),
        WorkloadType::Lrange300 => create_lrange_template(key_prefix, key_width, 300),
        WorkloadType::Lrange500 => create_lrange_template(key_prefix, key_width, 500),
        WorkloadType::Lrange600 => create_lrange_template(key_prefix, key_width, 600),

        // === Set commands ===
        WorkloadType::Sadd => CommandTemplate::new("SADD")
            .arg_str("SADD")
            .arg_prefixed_key(key_prefix, key_width)
            .arg_rand_int(key_width),

        WorkloadType::Spop => CommandTemplate::new("SPOP")
            .arg_str("SPOP")
            .arg_prefixed_key(key_prefix, key_width),

        // === Hash commands ===
        WorkloadType::Hset => CommandTemplate::new("HSET")
            .arg_str("HSET")
            .arg_prefixed_key(key_prefix, key_width)
            .arg_str("field")
            .arg_literal(&vec![b'x'; data_size]),

        // === Sorted set commands ===
        WorkloadType::Zadd => CommandTemplate::new("ZADD")
            .arg_str("ZADD")
            .arg_prefixed_key(key_prefix, key_width)
            .arg_rand_int(key_width) // score
            .arg_str("member"),

        WorkloadType::Zpopmin => CommandTemplate::new("ZPOPMIN")
            .arg_str("ZPOPMIN")
            .arg_prefixed_key(key_prefix, key_width),

        // === Multi-key commands ===
        WorkloadType::Mset => create_mset_template(key_prefix, key_width, data_size, 10),

        // === Vector search commands ===
        WorkloadType::VecLoad => {
            let sc = search_config.expect("VecLoad requires search config");
            create_vec_load_template(&sc.prefix, key_width, &sc.vector_field, sc.vec_byte_len())
        }

        WorkloadType::VecQuery => {
            let sc = search_config.expect("VecQuery requires search config");
            create_vec_query_template(sc)
        }

        WorkloadType::VecDelete => {
            let sc = search_config.expect("VecDelete requires search config");
            CommandTemplate::new("DEL")
                .arg_str("DEL")
                .arg_prefixed_key(&sc.prefix, key_width)
        }

        WorkloadType::VecUpdate => {
            let sc = search_config.expect("VecUpdate requires search config");
            create_vec_load_template(&sc.prefix, key_width, &sc.vector_field, sc.vec_byte_len())
        }

        WorkloadType::Custom => {
            // Custom commands should be handled separately
            CommandTemplate::new("CUSTOM").arg_str("PING")
        }
    }
}

/// Create LRANGE template with specified count
fn create_lrange_template(key_prefix: &str, key_width: usize, count: i32) -> CommandTemplate {
    CommandTemplate::new(&format!("LRANGE_{}", count))
        .arg_str("LRANGE")
        .arg_prefixed_key(key_prefix, key_width)
        .arg_str("0")
        .arg_str(&(count - 1).to_string())
}

/// Create MSET template with multiple keys
fn create_mset_template(
    key_prefix: &str,
    key_width: usize,
    data_size: usize,
    num_keys: usize,
) -> CommandTemplate {
    let mut template = CommandTemplate::new("MSET").arg_str("MSET");

    for _ in 0..num_keys {
        template = template
            .arg_prefixed_key(key_prefix, key_width)
            .arg_literal(&vec![b'x'; data_size]);
    }

    template
}

/// Create HSET template for vector loading
fn create_vec_load_template(
    prefix: &str,
    key_width: usize,
    vector_field: &str,
    vec_byte_len: usize,
) -> CommandTemplate {
    CommandTemplate::new("HSET")
        .arg_str("HSET")
        .arg_prefixed_key(prefix, key_width)
        .arg_str(vector_field)
        .arg_vector(vec_byte_len)
}

/// Create FT.SEARCH template for vector queries
fn create_vec_query_template(search_config: &SearchConfig) -> CommandTemplate {
    // Build query string: "*=>[KNN $K @embedding $BLOB]"
    // For the template, we use a placeholder for the vector blob

    let mut template = CommandTemplate::new("FT.SEARCH")
        .arg_str("FT.SEARCH")
        .arg_str(&search_config.index_name);

    // Build query string based on config
    let query = format!(
        "*=>[KNN {} @{} $BLOB{}]",
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
        .arg_query_vector(search_config.vec_byte_len()); // Use query vector for FT.SEARCH

    if search_config.nocontent {
        template = template.arg_str("NOCONTENT");
    }

    template
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DistanceMetric, VectorAlgorithm};

    #[test]
    fn test_create_ping_template() {
        let template = create_template(WorkloadType::Ping, "key:", 3, None);
        let buf = template.build(1);
        assert!(buf.placeholders[0].is_empty());
    }

    #[test]
    fn test_create_set_template() {
        let template = create_template(WorkloadType::Set, "key:", 100, None);
        let buf = template.build(1);
        assert_eq!(buf.placeholders[0].len(), 1); // Key placeholder
    }

    #[test]
    fn test_create_get_template() {
        let template = create_template(WorkloadType::Get, "key:", 3, None);
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
        };

        let template = create_template(WorkloadType::VecLoad, "key:", 3, Some(&search_config));
        let buf = template.build(1);

        // Should have key and vector placeholders
        assert_eq!(buf.placeholders[0].len(), 2);
    }
}
