//! Event-driven benchmark worker using mio for non-blocking I/O
//!
//! This implementation matches the C valkey-benchmark's architecture:
//! - Non-blocking sockets
//! - Event loop (mio::Poll) similar to ae_event_loop
//! - Multiple clients multiplexed on single thread
//! - Write when writable, read when readable

use std::collections::VecDeque;
use std::io::{self, ErrorKind, Read, Write};
use std::net::TcpStream;
use std::os::unix::io::AsRawFd;
use std::time::{Duration, Instant};

use hdrhistogram::Histogram;
use mio::event::Event;
use mio::net::TcpStream as MioTcpStream;
use mio::{Events, Interest, Poll, Token};

use super::counters::GlobalCounters;
use super::worker::RecallStats;
use crate::client::{CommandBuffer, PlaceholderOffset, PlaceholderType};
use crate::config::BenchmarkConfig;
use crate::dataset::DatasetContext;
use crate::utils::{RespDecoder, RespValue};
use std::sync::Arc;

/// Client state in the event loop
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClientState {
    /// Ready to send a new request
    Idle,
    /// Writing request to socket
    Writing,
    /// Waiting for response
    Reading,
}

/// Event-driven client
struct EventClient {
    /// Mio TCP stream (non-blocking)
    stream: MioTcpStream,
    /// Token for mio registry
    token: Token,
    /// Current state
    state: ClientState,
    /// Write buffer (command bytes)
    write_buf: Vec<u8>,
    /// Write position
    write_pos: usize,
    /// Read buffer
    read_buf: Vec<u8>,
    /// Read position
    read_pos: usize,
    /// Number of responses expected
    pending_responses: usize,
    /// Responses received
    responses: Vec<RespValue>,
    /// Request start time (for latency)
    start_time: Option<Instant>,
    /// Pipeline size
    pipeline: usize,
    /// Inflight dataset indices
    inflight_indices: VecDeque<u64>,
    /// Query indices for recall
    query_indices: VecDeque<u64>,
}

impl EventClient {
    fn new(stream: MioTcpStream, token: Token, write_buf: Vec<u8>, pipeline: usize) -> Self {
        Self {
            stream,
            token,
            state: ClientState::Idle,
            write_buf,
            write_pos: 0,
            read_buf: vec![0u8; 65536],
            read_pos: 0,
            pending_responses: 0,
            responses: Vec::with_capacity(pipeline),
            start_time: None,
            pipeline,
            inflight_indices: VecDeque::with_capacity(pipeline),
            query_indices: VecDeque::with_capacity(pipeline),
        }
    }

    /// Start a new request
    fn start_request(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.pending_responses = self.pipeline;
        self.responses.clear();
        self.start_time = Some(Instant::now());
        self.state = ClientState::Writing;
    }

    /// Try to write as much as possible (non-blocking)
    /// Returns: Ok(true) if write complete, Ok(false) if would block, Err on error
    fn try_write(&mut self) -> io::Result<bool> {
        while self.write_pos < self.write_buf.len() {
            match self.stream.write(&self.write_buf[self.write_pos..]) {
                Ok(0) => {
                    return Err(io::Error::new(ErrorKind::WriteZero, "Connection closed"));
                }
                Ok(n) => {
                    self.write_pos += n;
                }
                Err(ref e) if e.kind() == ErrorKind::WouldBlock => {
                    return Ok(false); // Would block, need to wait
                }
                Err(ref e) if e.kind() == ErrorKind::Interrupted => {
                    continue; // Retry
                }
                Err(e) => return Err(e),
            }
        }
        // Write complete, switch to reading
        self.state = ClientState::Reading;
        Ok(true)
    }

    /// Try to read responses (non-blocking)
    /// Returns: Ok(true) if all responses received, Ok(false) if need more data, Err on error
    fn try_read(&mut self) -> io::Result<bool> {
        // Read available data into buffer
        loop {
            // Ensure buffer has space
            if self.read_pos >= self.read_buf.len() {
                self.read_buf.resize(self.read_buf.len() * 2, 0);
            }

            match self.stream.read(&mut self.read_buf[self.read_pos..]) {
                Ok(0) => {
                    return Err(io::Error::new(
                        ErrorKind::UnexpectedEof,
                        "Connection closed",
                    ));
                }
                Ok(n) => {
                    self.read_pos += n;
                }
                Err(ref e) if e.kind() == ErrorKind::WouldBlock => {
                    break; // No more data available
                }
                Err(ref e) if e.kind() == ErrorKind::Interrupted => {
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        // Try to parse responses
        self.parse_responses()
    }

    /// Parse RESP responses from read buffer
    fn parse_responses(&mut self) -> io::Result<bool> {
        let mut parse_pos = 0;

        while self.responses.len() < self.pending_responses {
            let data = &self.read_buf[parse_pos..self.read_pos];
            if data.is_empty() {
                break;
            }

            match parse_resp_value(data) {
                Ok((value, consumed)) => {
                    self.responses.push(value);
                    parse_pos += consumed;
                }
                Err(ParseError::Incomplete) => {
                    break; // Need more data
                }
                Err(ParseError::Invalid(msg)) => {
                    return Err(io::Error::new(ErrorKind::InvalidData, msg));
                }
            }
        }

        // Shift remaining data to front of buffer
        if parse_pos > 0 && parse_pos < self.read_pos {
            self.read_buf.copy_within(parse_pos..self.read_pos, 0);
            self.read_pos -= parse_pos;
        } else if parse_pos == self.read_pos {
            self.read_pos = 0;
        }

        // Check if all responses received
        if self.responses.len() >= self.pending_responses {
            self.state = ClientState::Idle;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get latency in microseconds
    fn latency_us(&self) -> u64 {
        self.start_time
            .map(|t| t.elapsed().as_micros() as u64)
            .unwrap_or(0)
    }
}

/// Parse error
enum ParseError {
    Incomplete,
    Invalid(String),
}

/// Parse a single RESP value from bytes
/// Returns (value, bytes_consumed) or error
fn parse_resp_value(data: &[u8]) -> Result<(RespValue, usize), ParseError> {
    if data.is_empty() {
        return Err(ParseError::Incomplete);
    }

    match data[0] {
        b'+' => parse_simple_string(data),
        b'-' => parse_error(data),
        b':' => parse_integer(data),
        b'$' => parse_bulk_string(data),
        b'*' => parse_array(data),
        _ => Err(ParseError::Invalid(format!(
            "Invalid RESP type byte: {}",
            data[0]
        ))),
    }
}

fn find_crlf(data: &[u8]) -> Option<usize> {
    data.windows(2).position(|w| w == b"\r\n")
}

fn parse_simple_string(data: &[u8]) -> Result<(RespValue, usize), ParseError> {
    let crlf = find_crlf(data).ok_or(ParseError::Incomplete)?;
    let s = String::from_utf8_lossy(&data[1..crlf]).to_string();
    Ok((RespValue::SimpleString(s), crlf + 2))
}

fn parse_error(data: &[u8]) -> Result<(RespValue, usize), ParseError> {
    let crlf = find_crlf(data).ok_or(ParseError::Incomplete)?;
    let s = String::from_utf8_lossy(&data[1..crlf]).to_string();
    Ok((RespValue::Error(s), crlf + 2))
}

fn parse_integer(data: &[u8]) -> Result<(RespValue, usize), ParseError> {
    let crlf = find_crlf(data).ok_or(ParseError::Incomplete)?;
    let s = std::str::from_utf8(&data[1..crlf])
        .map_err(|_| ParseError::Invalid("Invalid integer".to_string()))?;
    let n: i64 = s
        .parse()
        .map_err(|_| ParseError::Invalid("Invalid integer".to_string()))?;
    Ok((RespValue::Integer(n), crlf + 2))
}

fn parse_bulk_string(data: &[u8]) -> Result<(RespValue, usize), ParseError> {
    let crlf = find_crlf(data).ok_or(ParseError::Incomplete)?;
    let len_str = std::str::from_utf8(&data[1..crlf])
        .map_err(|_| ParseError::Invalid("Invalid bulk string length".to_string()))?;
    let len: i64 = len_str
        .parse()
        .map_err(|_| ParseError::Invalid("Invalid bulk string length".to_string()))?;

    if len < 0 {
        return Ok((RespValue::Null, crlf + 2));
    }

    let len = len as usize;
    let total_len = crlf + 2 + len + 2;

    if data.len() < total_len {
        return Err(ParseError::Incomplete);
    }

    let content = data[crlf + 2..crlf + 2 + len].to_vec();
    Ok((RespValue::BulkString(content), total_len))
}

fn parse_array(data: &[u8]) -> Result<(RespValue, usize), ParseError> {
    let crlf = find_crlf(data).ok_or(ParseError::Incomplete)?;
    let count_str = std::str::from_utf8(&data[1..crlf])
        .map_err(|_| ParseError::Invalid("Invalid array count".to_string()))?;
    let count: i64 = count_str
        .parse()
        .map_err(|_| ParseError::Invalid("Invalid array count".to_string()))?;

    if count < 0 {
        return Ok((RespValue::Null, crlf + 2));
    }

    let mut pos = crlf + 2;
    let mut elements = Vec::with_capacity(count as usize);

    for _ in 0..count {
        if pos >= data.len() {
            return Err(ParseError::Incomplete);
        }
        let (elem, consumed) = parse_resp_value(&data[pos..])?;
        elements.push(elem);
        pos += consumed;
    }

    Ok((RespValue::Array(elements), pos))
}

/// Result from event worker
pub struct EventWorkerResult {
    pub worker_id: usize,
    pub histogram: Histogram<u64>,
    pub recall_stats: RecallStats,
    pub redirect_count: u64,
    pub topology_refresh_count: u64,
    pub error_count: u64,
    pub requests_processed: u64,
}

/// Event-driven benchmark worker
pub struct EventWorker {
    id: usize,
    poll: Poll,
    clients: Vec<EventClient>,
    events: Events,
    rng: fastrand::Rng,
    histogram: Histogram<u64>,
    recall_stats: RecallStats,
    pipeline: usize,
    keyspace_len: u64,
    sequential: bool,
    command_template: CommandBuffer,
    key_prefix: String,
    k: usize,
    compute_recall: bool,
    error_count: u64,
    requests_processed: u64,
}

impl EventWorker {
    /// Create new event-driven worker
    pub fn new(
        id: usize,
        addresses: Vec<(String, u16)>,
        clients_per_addr: usize,
        config: &BenchmarkConfig,
        command_template: CommandBuffer,
    ) -> io::Result<Self> {
        let poll = Poll::new()?;
        let mut clients = Vec::new();
        let mut token_counter = 0usize;

        // Create clients distributed across addresses
        for (addr_idx, (host, port)) in addresses.iter().enumerate() {
            for _ in 0..clients_per_addr {
                let addr = format!("{}:{}", host, port);
                let std_stream = std::net::TcpStream::connect(&addr)?;
                std_stream.set_nonblocking(true)?;
                std_stream.set_nodelay(true)?;

                let mut mio_stream = MioTcpStream::from_std(std_stream);
                let token = Token(token_counter);
                token_counter += 1;

                // Register with poll - initially interested in WRITABLE
                poll.registry().register(
                    &mut mio_stream,
                    token,
                    Interest::READABLE | Interest::WRITABLE,
                )?;

                let client = EventClient::new(
                    mio_stream,
                    token,
                    command_template.bytes.clone(),
                    config.pipeline as usize,
                );
                clients.push(client);
            }
        }

        // Initialize RNG
        let seed = if config.seed == 0 {
            fastrand::u64(..)
        } else {
            config.seed.wrapping_add(id as u64)
        };
        let rng = fastrand::Rng::with_seed(seed);

        // Histogram
        let histogram =
            Histogram::new_with_bounds(1, 3_600_000_000, 3).expect("Failed to create histogram");

        // Recall config
        let (compute_recall, key_prefix, k) = if let Some(ref sc) = config.search_config {
            (true, sc.prefix.clone(), sc.k as usize)
        } else {
            (false, String::new(), 0)
        };

        Ok(Self {
            id,
            poll,
            clients,
            events: Events::with_capacity(1024),
            rng,
            histogram,
            recall_stats: RecallStats::new(),
            pipeline: config.pipeline as usize,
            keyspace_len: config.keyspace_len,
            sequential: config.sequential,
            command_template,
            key_prefix,
            k,
            compute_recall,
            error_count: 0,
            requests_processed: 0,
        })
    }

    /// Run the event loop
    pub fn run(
        mut self,
        counters: Arc<GlobalCounters>,
        dataset: Option<Arc<DatasetContext>>,
    ) -> EventWorkerResult {
        let batch_size = self.pipeline as u64;

        // Start all clients
        for client_idx in 0..self.clients.len() {
            if counters
                .claim_batch(batch_size)
                .is_some()
            {
                self.fill_placeholders_for(client_idx, &counters, dataset.as_deref());
                self.clients[client_idx].start_request();
            }
        }

        // Event loop - poll for I/O readiness
        loop {
            if counters.is_shutdown() || counters.is_duration_exceeded() {
                break;
            }

            // Check if all requests have been issued and all clients are idle
            if counters.is_complete() {
                let all_idle = self
                    .clients
                    .iter()
                    .all(|c| c.state == ClientState::Idle);
                if all_idle {
                    break;
                }
            }

            // Poll for events - use short timeout to also handle idle clients
            if self
                .poll
                .poll(&mut self.events, Some(Duration::from_micros(100)))
                .is_err()
            {
                continue;
            }

            // Process all clients - not just those with events
            // This is more like C's approach where we try operations optimistically
            for client_idx in 0..self.clients.len() {
                if counters.is_duration_exceeded() {
                    break;
                }

                let state = self.clients[client_idx].state;

                match state {
                    ClientState::Idle => {
                        // Nothing to do for idle - handled separately below
                    }
                    ClientState::Writing => {
                        // Try to write (optimistically)
                        match self.clients[client_idx].try_write() {
                            Ok(true) => {
                                // Write complete, now in Reading state
                            }
                            Ok(false) => {
                                // Would block, will try again next iteration
                            }
                            Err(_e) => {
                                self.error_count += batch_size;
                                self.clients[client_idx].state = ClientState::Idle;
                            }
                        }
                    }
                    ClientState::Reading => {
                        // Try to read (optimistically)
                        match self.clients[client_idx].try_read() {
                            Ok(true) => {
                                // All responses received
                                let latency = self.clients[client_idx].latency_us();
                                self.histogram.record(latency).ok();
                                self.requests_processed +=
                                    self.clients[client_idx].responses.len() as u64;
                                counters.record_finished(
                                    self.clients[client_idx].responses.len() as u64,
                                );

                                // Start next request immediately
                                if counters
                                    .claim_batch(batch_size)
                                    .is_some()
                                {
                                    self.fill_placeholders_for(
                                        client_idx,
                                        &counters,
                                        dataset.as_deref(),
                                    );
                                    self.clients[client_idx].start_request();
                                    // Immediately try to write (socket likely writable)
                                    let _ = self.clients[client_idx].try_write();
                                } else {
                                    self.clients[client_idx].state = ClientState::Idle;
                                }
                            }
                            Ok(false) => {
                                // Need more data, will try again next iteration
                            }
                            Err(_e) => {
                                self.error_count += batch_size;
                                self.clients[client_idx].state = ClientState::Idle;
                            }
                        }
                    }
                }
            }

            // Start idle clients
            for client_idx in 0..self.clients.len() {
                if self.clients[client_idx].state == ClientState::Idle {
                    if counters.is_duration_exceeded() {
                        break;
                    }
                    if counters
                        .claim_batch(batch_size)
                        .is_some()
                    {
                        self.fill_placeholders_for(client_idx, &counters, dataset.as_deref());
                        self.clients[client_idx].start_request();
                        // Immediately try to write
                        let _ = self.clients[client_idx].try_write();
                    }
                }
            }
        }

        EventWorkerResult {
            worker_id: self.id,
            histogram: self.histogram,
            recall_stats: self.recall_stats,
            redirect_count: 0,
            topology_refresh_count: 0,
            error_count: self.error_count,
            requests_processed: self.requests_processed,
        }
    }

    /// Fill placeholders for a specific client by index
    fn fill_placeholders_for(
        &mut self,
        client_idx: usize,
        counters: &GlobalCounters,
        dataset: Option<&DatasetContext>,
    ) {
        // Fill placeholders in write buffer
        for (cmd_idx, phs) in self.command_template.placeholders.iter().enumerate() {
            for ph in phs {
                let offset = self.command_template.absolute_offset(cmd_idx, ph.offset);
                match ph.placeholder_type {
                    PlaceholderType::Key => {
                        if dataset.is_some() {
                            continue; // Set with Vector
                        }
                        let key = if self.sequential {
                            counters.next_seq_key(self.keyspace_len)
                        } else {
                            self.rng.u64(0..self.keyspace_len)
                        };
                        write_fixed_width_u64(&mut self.clients[client_idx].write_buf, offset, key, ph.len);
                    }
                    PlaceholderType::Vector => {
                        if let Some(ds) = dataset {
                            let idx = counters.next_dataset_idx() % ds.num_vectors();
                            let vec_bytes = ds.get_vector_bytes(idx);
                            self.clients[client_idx].write_buf[offset..offset + vec_bytes.len()]
                                .copy_from_slice(vec_bytes);
                            // Set key to match
                            for ph2 in phs {
                                if matches!(ph2.placeholder_type, PlaceholderType::Key) {
                                    let key_offset =
                                        self.command_template.absolute_offset(cmd_idx, ph2.offset);
                                    write_fixed_width_u64(
                                        &mut self.clients[client_idx].write_buf,
                                        key_offset,
                                        idx,
                                        ph2.len,
                                    );
                                    break;
                                }
                            }
                        }
                    }
                    PlaceholderType::QueryVector => {
                        if let Some(ds) = dataset {
                            let idx = self.rng.u64(0..ds.num_queries());
                            let vec_bytes = ds.get_query_bytes(idx);
                            self.clients[client_idx].write_buf[offset..offset + vec_bytes.len()]
                                .copy_from_slice(vec_bytes);
                        }
                    }
                    PlaceholderType::RandInt => {
                        let value = self.rng.u64(..);
                        write_fixed_width_u64(&mut self.clients[client_idx].write_buf, offset, value, ph.len);
                    }
                    PlaceholderType::ClusterTag => {
                        // Static tag for now
                        self.clients[client_idx].write_buf[offset..offset + 5].copy_from_slice(b"{000}");
                    }
                }
            }
        }
    }
}

/// Write u64 as fixed-width decimal (zero-padded)
#[inline]
fn write_fixed_width_u64(buf: &mut [u8], offset: usize, value: u64, width: usize) {
    let mut v = value;
    for i in (0..width).rev() {
        buf[offset + i] = b'0' + (v % 10) as u8;
        v /= 10;
    }
}
