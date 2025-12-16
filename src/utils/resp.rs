//! RESP (Redis Serialization Protocol) encoder and decoder
//!
//! This module provides zero-copy RESP encoding for commands and
//! streaming RESP decoding for responses.

use std::io::{self, BufRead};

/// RESP value types
#[derive(Debug, Clone, PartialEq)]
pub enum RespValue {
    /// Simple string (+OK\r\n)
    SimpleString(String),
    /// Error (-ERR message\r\n)
    Error(String),
    /// Integer (:1000\r\n)
    Integer(i64),
    /// Bulk string ($6\r\nfoobar\r\n)
    BulkString(Vec<u8>),
    /// Null bulk string ($-1\r\n)
    Null,
    /// Array (*2\r\n...)
    Array(Vec<RespValue>),
}

impl RespValue {
    /// Check if this is an error response
    pub fn is_error(&self) -> bool {
        matches!(self, RespValue::Error(_))
    }

    /// Check if this is a MOVED error
    pub fn is_moved(&self) -> bool {
        match self {
            RespValue::Error(e) => e.starts_with("MOVED"),
            _ => false,
        }
    }

    /// Check if this is an ASK error
    pub fn is_ask(&self) -> bool {
        match self {
            RespValue::Error(e) => e.starts_with("ASK"),
            _ => false,
        }
    }

    /// Parse MOVED/ASK error to extract slot and target
    /// Returns (slot, host, port)
    pub fn parse_redirect(&self) -> Option<(u16, String, u16)> {
        match self {
            RespValue::Error(e) => {
                let parts: Vec<&str> = e.split_whitespace().collect();
                if parts.len() >= 3 && (parts[0] == "MOVED" || parts[0] == "ASK") {
                    let slot: u16 = parts[1].parse().ok()?;
                    let addr_parts: Vec<&str> = parts[2].split(':').collect();
                    if addr_parts.len() == 2 {
                        let host = addr_parts[0].to_string();
                        let port: u16 = addr_parts[1].parse().ok()?;
                        return Some((slot, host, port));
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Get as string (for simple string or bulk string)
    pub fn as_str(&self) -> Option<&str> {
        match self {
            RespValue::SimpleString(s) => Some(s),
            RespValue::BulkString(b) => std::str::from_utf8(b).ok(),
            _ => None,
        }
    }

    /// Get as bytes (for bulk string)
    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            RespValue::BulkString(b) => Some(b),
            _ => None,
        }
    }

    /// Get as integer
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            RespValue::Integer(i) => Some(*i),
            _ => None,
        }
    }

    /// Get as array
    pub fn as_array(&self) -> Option<&[RespValue]> {
        match self {
            RespValue::Array(arr) => Some(arr),
            _ => None,
        }
    }
}

/// RESP encoder with pre-allocated buffer
pub struct RespEncoder {
    buf: Vec<u8>,
}

impl RespEncoder {
    /// Create new encoder with specified capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
        }
    }

    /// Clear buffer for reuse
    pub fn clear(&mut self) {
        self.buf.clear();
    }

    /// Get encoded bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    /// Take ownership of buffer
    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }

    /// Get mutable reference to internal buffer
    pub fn buffer_mut(&mut self) -> &mut Vec<u8> {
        &mut self.buf
    }

    /// Encode a command as RESP array
    /// Each argument is encoded as a bulk string
    pub fn encode_command(&mut self, args: &[&[u8]]) {
        // Array header: *<count>\r\n
        self.buf.push(b'*');
        self.write_int(args.len() as i64);
        self.buf.extend_from_slice(b"\r\n");

        // Each argument as bulk string: $<len>\r\n<data>\r\n
        for arg in args {
            self.buf.push(b'$');
            self.write_int(arg.len() as i64);
            self.buf.extend_from_slice(b"\r\n");
            self.buf.extend_from_slice(arg);
            self.buf.extend_from_slice(b"\r\n");
        }
    }

    /// Encode a command from string slices
    pub fn encode_command_str(&mut self, args: &[&str]) {
        let byte_args: Vec<&[u8]> = args.iter().map(|s| s.as_bytes()).collect();
        self.encode_command(&byte_args);
    }

    /// Encode multiple commands (pipeline)
    pub fn encode_pipeline(&mut self, commands: &[Vec<&[u8]>]) {
        for cmd in commands {
            self.encode_command(cmd);
        }
    }

    /// Write integer using fast itoa
    #[inline]
    fn write_int(&mut self, value: i64) {
        let mut buffer = itoa::Buffer::new();
        let s = buffer.format(value);
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// Encode inline command (space-separated, for simple commands)
    pub fn encode_inline(&mut self, args: &[&str]) {
        for (i, arg) in args.iter().enumerate() {
            if i > 0 {
                self.buf.push(b' ');
            }
            self.buf.extend_from_slice(arg.as_bytes());
        }
        self.buf.extend_from_slice(b"\r\n");
    }
}

/// RESP decoder for streaming reads
pub struct RespDecoder<R> {
    reader: R,
    line_buf: String,
}

impl<R: BufRead> RespDecoder<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            line_buf: String::with_capacity(256),
        }
    }

    /// Decode next RESP value from stream
    pub fn decode(&mut self) -> io::Result<RespValue> {
        // Read type byte
        self.line_buf.clear();
        self.reader.read_line(&mut self.line_buf)?;

        if self.line_buf.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Connection closed",
            ));
        }

        let line = self.line_buf.trim_end_matches(&['\r', '\n'][..]);
        if line.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Empty RESP line",
            ));
        }

        let type_byte = line.as_bytes()[0];
        let content = &line[1..];

        match type_byte {
            b'+' => Ok(RespValue::SimpleString(content.to_string())),
            b'-' => Ok(RespValue::Error(content.to_string())),
            b':' => {
                let value: i64 = content
                    .parse()
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid integer"))?;
                Ok(RespValue::Integer(value))
            }
            b'$' => {
                let len: i64 = content.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid bulk string length")
                })?;

                if len < 0 {
                    return Ok(RespValue::Null);
                }

                let len = len as usize;
                let mut data = vec![0u8; len];
                self.reader.read_exact(&mut data)?;

                // Read trailing \r\n
                let mut crlf = [0u8; 2];
                self.reader.read_exact(&mut crlf)?;

                Ok(RespValue::BulkString(data))
            }
            b'*' => {
                let count: i64 = content.parse().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "Invalid array length")
                })?;

                if count < 0 {
                    return Ok(RespValue::Null);
                }

                let mut elements = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    elements.push(self.decode()?);
                }

                Ok(RespValue::Array(elements))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid RESP type byte: {}", type_byte as char),
            )),
        }
    }

    /// Decode multiple responses (for pipeline)
    pub fn decode_pipeline(&mut self, count: usize) -> io::Result<Vec<RespValue>> {
        let mut responses = Vec::with_capacity(count);
        for _ in 0..count {
            responses.push(self.decode()?);
        }
        Ok(responses)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_encode_simple_command() {
        let mut encoder = RespEncoder::with_capacity(64);
        encoder.encode_command_str(&["PING"]);
        assert_eq!(encoder.as_bytes(), b"*1\r\n$4\r\nPING\r\n");
    }

    #[test]
    fn test_encode_set_command() {
        let mut encoder = RespEncoder::with_capacity(64);
        encoder.encode_command_str(&["SET", "key", "value"]);
        assert_eq!(
            encoder.as_bytes(),
            b"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n"
        );
    }

    #[test]
    fn test_decode_simple_string() {
        let data = b"+OK\r\n";
        let mut decoder = RespDecoder::new(Cursor::new(&data[..]));
        let value = decoder.decode().unwrap();
        assert_eq!(value, RespValue::SimpleString("OK".to_string()));
    }

    #[test]
    fn test_decode_error() {
        let data = b"-ERR unknown command\r\n";
        let mut decoder = RespDecoder::new(Cursor::new(&data[..]));
        let value = decoder.decode().unwrap();
        assert_eq!(value, RespValue::Error("ERR unknown command".to_string()));
    }

    #[test]
    fn test_decode_integer() {
        let data = b":1000\r\n";
        let mut decoder = RespDecoder::new(Cursor::new(&data[..]));
        let value = decoder.decode().unwrap();
        assert_eq!(value, RespValue::Integer(1000));
    }

    #[test]
    fn test_decode_bulk_string() {
        let data = b"$6\r\nfoobar\r\n";
        let mut decoder = RespDecoder::new(Cursor::new(&data[..]));
        let value = decoder.decode().unwrap();
        assert_eq!(value, RespValue::BulkString(b"foobar".to_vec()));
    }

    #[test]
    fn test_decode_array() {
        let data = b"*2\r\n$3\r\nfoo\r\n$3\r\nbar\r\n";
        let mut decoder = RespDecoder::new(Cursor::new(&data[..]));
        let value = decoder.decode().unwrap();
        assert_eq!(
            value,
            RespValue::Array(vec![
                RespValue::BulkString(b"foo".to_vec()),
                RespValue::BulkString(b"bar".to_vec()),
            ])
        );
    }

    #[test]
    fn test_parse_moved() {
        let value = RespValue::Error("MOVED 3999 127.0.0.1:7001".to_string());
        let (slot, host, port) = value.parse_redirect().unwrap();
        assert_eq!(slot, 3999);
        assert_eq!(host, "127.0.0.1");
        assert_eq!(port, 7001);
    }

    #[test]
    fn test_decode_null() {
        let data = b"$-1\r\n";
        let mut decoder = RespDecoder::new(Cursor::new(&data[..]));
        let value = decoder.decode().unwrap();
        assert_eq!(value, RespValue::Null);
    }
}
