//! Zero-copy, `unsafe`-free reader/writer for the `.hss` hybrid inference-state
//! container.
//!
//! On-disk layout (deliberately safetensors-shaped — "safetensors for inference
//! state"):
//!
//! ```text
//!   [ 8 bytes : u64 little-endian header length N ]
//!   [ N bytes : UTF-8 JSON header                 ]
//!   [ remainder : raw tensor data bytes           ]
//! ```
//!
//! The JSON header maps each tensor `name` to
//! `{ "dtype": "F32", "shape": [..], "data_offsets": [start, end] }`, plus a
//! reserved `"__metadata__"` object of `string -> string` pairs that carries the
//! hybrid-state semantics (per-layer ROLE, chunk-boundary, `_seen_tokens`,
//! conv-phase). Offsets are relative to the start of the data section.
//!
//! This crate owns the *untrusted-input boundary*: every length and offset read
//! off disk is bounds-checked with checked arithmetic before any slice is
//! produced, and the crate contains no `unsafe` (`#![forbid(unsafe_code)]` via
//! the manifest lint). Higher-level semantics live in the Python layer and in
//! `hss-spec/hss-spec.md`; this crate only guarantees the bytes are well-formed.

use std::collections::BTreeMap;
use std::str::FromStr;

use serde_json::{Map, Value};

/// Container spec version this crate reads/writes.
pub const HSS_VERSION: &str = "0.1";

/// Defensive upper bound on the JSON header length (untrusted input).
pub const MAX_HEADER_LEN: u64 = 100 * 1024 * 1024; // 100 MiB

/// Errors produced while parsing or serializing a `.hss` container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HssError {
    /// Buffer is shorter than the 8-byte length prefix, or shorter than the
    /// declared header length.
    TooShort,
    /// `8 + header_len` overflowed `u64`.
    HeaderLenOverflow,
    /// Declared header length exceeds [`MAX_HEADER_LEN`].
    HeaderTooLarge(u64),
    /// Header bytes were not valid UTF-8.
    HeaderUtf8,
    /// Header was not valid JSON.
    HeaderJson(String),
    /// `__metadata__` was present but was not an object of string -> string.
    BadMetadata,
    /// Tensor dtype string was not one of the known dtypes.
    UnknownDtype(String),
    /// A tensor entry was structurally malformed.
    BadTensorEntry(String),
    /// A tensor's `data_offsets` end was past the data section.
    OffsetOutOfBounds {
        name: String,
        end: u64,
        data_len: usize,
    },
    /// A tensor's `data_offsets` had `start > end`.
    OffsetOrder { name: String },
    /// The product of `shape * dtype_size` overflowed `u64`.
    ShapeOverflow { name: String },
    /// `shape * dtype_size` did not match the byte length implied by offsets.
    LengthMismatch {
        name: String,
        expected: u64,
        got: u64,
    },
    /// `__metadata__` is a reserved name and cannot be used as a tensor name.
    ReservedName,
}

impl std::fmt::Display for HssError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HssError::TooShort => write!(f, "buffer too short for declared header"),
            HssError::HeaderLenOverflow => write!(f, "header length overflowed"),
            HssError::HeaderTooLarge(n) => {
                write!(f, "header length {n} exceeds maximum {MAX_HEADER_LEN}")
            }
            HssError::HeaderUtf8 => write!(f, "header is not valid UTF-8"),
            HssError::HeaderJson(e) => write!(f, "header is not valid JSON: {e}"),
            HssError::BadMetadata => {
                write!(f, "__metadata__ must be an object of string to string")
            }
            HssError::UnknownDtype(d) => write!(f, "unknown dtype: {d}"),
            HssError::BadTensorEntry(n) => write!(f, "malformed tensor entry: {n}"),
            HssError::OffsetOutOfBounds {
                name,
                end,
                data_len,
            } => {
                write!(
                    f,
                    "tensor {name}: data_offsets end {end} past data section length {data_len}"
                )
            }
            HssError::OffsetOrder { name } => write!(f, "tensor {name}: data_offsets start > end"),
            HssError::ShapeOverflow { name } => {
                write!(f, "tensor {name}: shape product overflowed")
            }
            HssError::LengthMismatch {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "tensor {name}: shape implies {expected} bytes but offsets span {got}"
                )
            }
            HssError::ReservedName => write!(f, "__metadata__ is a reserved name"),
        }
    }
}

impl std::error::Error for HssError {}

/// Element data types. These are byte-width containers only; this crate does not
/// interpret the numeric values (e.g. BF16 and I16 are both 2-byte opaque).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Dtype {
    /// Size of one element in bytes.
    pub fn size(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 => 2,
            Dtype::I8 | Dtype::U8 | Dtype::Bool => 1,
        }
    }

    /// Canonical uppercase dtype string used in the header.
    pub fn as_str(self) -> &'static str {
        match self {
            Dtype::F64 => "F64",
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::I64 => "I64",
            Dtype::I32 => "I32",
            Dtype::I16 => "I16",
            Dtype::I8 => "I8",
            Dtype::U8 => "U8",
            Dtype::Bool => "BOOL",
        }
    }
}

impl std::str::FromStr for Dtype {
    type Err = HssError;

    /// Parse a dtype string (case-sensitive canonical form).
    fn from_str(s: &str) -> Result<Dtype, HssError> {
        Ok(match s {
            "F64" => Dtype::F64,
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            "I16" => Dtype::I16,
            "I8" => Dtype::I8,
            "U8" => Dtype::U8,
            "BOOL" => Dtype::Bool,
            other => return Err(HssError::UnknownDtype(other.to_string())),
        })
    }
}

/// Parsed description of one tensor within a container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// `[start, end)` byte offsets relative to the data section.
    pub data_offsets: [usize; 2],
}

/// A validated, borrowed view over a `.hss` buffer. All offsets have already
/// been bounds-checked, so [`HssView::tensor_bytes`] cannot index out of range.
#[derive(Debug, PartialEq, Eq)]
pub struct HssView<'a> {
    metadata: BTreeMap<String, String>,
    tensors: BTreeMap<String, TensorInfo>,
    data: &'a [u8],
}

impl<'a> HssView<'a> {
    /// The `__metadata__` string map.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Tensor names, sorted.
    pub fn names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Descriptor for a tensor by name.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Borrowed bytes for a tensor. Always in-bounds for any name present in the
    /// view (offsets validated at parse time).
    pub fn tensor_bytes(&self, name: &str) -> Option<&'a [u8]> {
        let t = self.tensors.get(name)?;
        Some(&self.data[t.data_offsets[0]..t.data_offsets[1]])
    }

    /// Iterate `(name, info)` in sorted name order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &TensorInfo)> {
        self.tensors.iter()
    }
}

/// Parse a `.hss` buffer, validating every offset against the data section.
///
/// Never panics on adversarial input: returns [`HssError`] instead.
pub fn parse(buf: &[u8]) -> Result<HssView<'_>, HssError> {
    if buf.len() < 8 {
        return Err(HssError::TooShort);
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&buf[0..8]);
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_LEN {
        return Err(HssError::HeaderTooLarge(header_len));
    }
    let header_end = 8u64
        .checked_add(header_len)
        .ok_or(HssError::HeaderLenOverflow)?;
    if header_end > buf.len() as u64 {
        return Err(HssError::TooShort);
    }
    let header_end = header_end as usize;
    let header_bytes = &buf[8..header_end];
    let data = &buf[header_end..];

    let header_str = std::str::from_utf8(header_bytes).map_err(|_| HssError::HeaderUtf8)?;
    let root: Map<String, Value> =
        serde_json::from_str(header_str).map_err(|e| HssError::HeaderJson(e.to_string()))?;

    let mut metadata = BTreeMap::new();
    let mut tensors = BTreeMap::new();
    for (key, value) in root {
        if key == "__metadata__" {
            let obj = value.as_object().ok_or(HssError::BadMetadata)?;
            for (mk, mv) in obj {
                let ms = mv.as_str().ok_or(HssError::BadMetadata)?;
                metadata.insert(mk.clone(), ms.to_string());
            }
            continue;
        }
        let info = parse_tensor_entry(&key, &value, data.len())?;
        tensors.insert(key, info);
    }

    Ok(HssView {
        metadata,
        tensors,
        data,
    })
}

fn parse_tensor_entry(name: &str, value: &Value, data_len: usize) -> Result<TensorInfo, HssError> {
    let obj = value
        .as_object()
        .ok_or_else(|| HssError::BadTensorEntry(name.to_string()))?;

    let dtype_s = obj
        .get("dtype")
        .and_then(Value::as_str)
        .ok_or_else(|| HssError::BadTensorEntry(name.to_string()))?;
    let dtype = Dtype::from_str(dtype_s)?;

    let shape_v = obj
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| HssError::BadTensorEntry(name.to_string()))?;
    let mut shape = Vec::with_capacity(shape_v.len());
    for d in shape_v {
        let u = d
            .as_u64()
            .ok_or_else(|| HssError::BadTensorEntry(name.to_string()))?;
        shape.push(u);
    }

    let off = obj
        .get("data_offsets")
        .and_then(Value::as_array)
        .ok_or_else(|| HssError::BadTensorEntry(name.to_string()))?;
    if off.len() != 2 {
        return Err(HssError::BadTensorEntry(name.to_string()));
    }
    let start = off[0]
        .as_u64()
        .ok_or_else(|| HssError::BadTensorEntry(name.to_string()))?;
    let end = off[1]
        .as_u64()
        .ok_or_else(|| HssError::BadTensorEntry(name.to_string()))?;

    if start > end {
        return Err(HssError::OffsetOrder {
            name: name.to_string(),
        });
    }
    if end > data_len as u64 {
        return Err(HssError::OffsetOutOfBounds {
            name: name.to_string(),
            end,
            data_len,
        });
    }

    let mut numel: u64 = 1;
    for &d in &shape {
        numel = numel
            .checked_mul(d)
            .ok_or_else(|| HssError::ShapeOverflow {
                name: name.to_string(),
            })?;
    }
    let expected =
        numel
            .checked_mul(dtype.size() as u64)
            .ok_or_else(|| HssError::ShapeOverflow {
                name: name.to_string(),
            })?;
    let span = end - start; // start <= end checked above
    if expected != span {
        return Err(HssError::LengthMismatch {
            name: name.to_string(),
            expected,
            got: span,
        });
    }

    // start <= end <= data_len <= isize::MAX in practice; safe to narrow.
    Ok(TensorInfo {
        dtype,
        shape,
        data_offsets: [start as usize, end as usize],
    })
}

/// An owned tensor to be serialized into a container.
#[derive(Debug, Clone)]
pub struct OwnedTensor {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

/// Serialize tensors plus a metadata map into a `.hss` byte buffer.
///
/// Tensors are written in sorted name order and the JSON header uses sorted
/// keys, so the output is deterministic (byte-identical) for identical input —
/// a prerequisite for the bitwise rehydration-equivalence contract.
pub fn serialize(
    tensors: &[OwnedTensor],
    metadata: &BTreeMap<String, String>,
) -> Result<Vec<u8>, HssError> {
    let mut sorted: Vec<&OwnedTensor> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));

    let mut header = Map::new();
    if !metadata.is_empty() {
        let mut m = Map::new();
        for (k, v) in metadata {
            m.insert(k.clone(), Value::String(v.clone()));
        }
        header.insert("__metadata__".to_string(), Value::Object(m));
    }

    let mut cursor: u64 = 0;
    for t in &sorted {
        if t.name == "__metadata__" {
            return Err(HssError::ReservedName);
        }
        let mut numel: u64 = 1;
        for &d in &t.shape {
            numel = numel
                .checked_mul(d)
                .ok_or_else(|| HssError::ShapeOverflow {
                    name: t.name.clone(),
                })?;
        }
        let expected =
            numel
                .checked_mul(t.dtype.size() as u64)
                .ok_or_else(|| HssError::ShapeOverflow {
                    name: t.name.clone(),
                })?;
        if expected != t.data.len() as u64 {
            return Err(HssError::LengthMismatch {
                name: t.name.clone(),
                expected,
                got: t.data.len() as u64,
            });
        }
        let start = cursor;
        let end =
            cursor
                .checked_add(t.data.len() as u64)
                .ok_or_else(|| HssError::ShapeOverflow {
                    name: t.name.clone(),
                })?;

        let mut entry = Map::new();
        entry.insert(
            "dtype".to_string(),
            Value::String(t.dtype.as_str().to_string()),
        );
        entry.insert(
            "shape".to_string(),
            Value::Array(t.shape.iter().map(|&d| Value::Number(d.into())).collect()),
        );
        entry.insert(
            "data_offsets".to_string(),
            Value::Array(vec![Value::Number(start.into()), Value::Number(end.into())]),
        );
        header.insert(t.name.clone(), Value::Object(entry));
        cursor = end;
    }

    let header_json = serde_json::to_vec(&Value::Object(header))
        .map_err(|e| HssError::HeaderJson(e.to_string()))?;

    let mut out = Vec::with_capacity(8 + header_json.len() + cursor as usize);
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_json);
    for t in &sorted {
        out.extend_from_slice(&t.data);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, dtype: Dtype, shape: &[u64], data: Vec<u8>) -> OwnedTensor {
        OwnedTensor {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            data,
        }
    }

    #[test]
    fn round_trip_with_metadata() {
        let mut meta = BTreeMap::new();
        meta.insert("hss_version".to_string(), HSS_VERSION.to_string());
        meta.insert("layer.0.role".to_string(), "recurrent_state".to_string());
        let tensors = vec![
            t("layer.0.recurrent", Dtype::F32, &[2, 3], vec![1u8; 24]),
            t("layer.1.conv", Dtype::F32, &[4], vec![2u8; 16]),
            t(
                "seen_tokens",
                Dtype::I64,
                &[1],
                vec![7, 0, 0, 0, 0, 0, 0, 0],
            ),
        ];
        let bytes = serialize(&tensors, &meta).unwrap();
        let view = parse(&bytes).unwrap();

        assert_eq!(
            view.metadata().get("hss_version").map(String::as_str),
            Some("0.1")
        );
        assert_eq!(
            view.metadata().get("layer.0.role").map(String::as_str),
            Some("recurrent_state")
        );
        assert_eq!(
            view.names(),
            vec!["layer.0.recurrent", "layer.1.conv", "seen_tokens"]
        );
        assert_eq!(view.tensor_bytes("layer.0.recurrent"), Some(&[1u8; 24][..]));
        assert_eq!(view.tensor_bytes("layer.1.conv"), Some(&[2u8; 16][..]));
        assert_eq!(view.info("seen_tokens").unwrap().dtype, Dtype::I64);
        assert_eq!(view.info("seen_tokens").unwrap().shape, vec![1]);
    }

    #[test]
    fn serialize_is_deterministic() {
        let tensors = vec![
            t("b", Dtype::U8, &[3], vec![9, 9, 9]),
            t("a", Dtype::U8, &[2], vec![1, 2]),
        ];
        let meta = BTreeMap::new();
        let one = serialize(&tensors, &meta).unwrap();
        let two = serialize(&tensors, &meta).unwrap();
        assert_eq!(one, two, "serialization must be byte-identical");
        // Reordering inputs must not change the output (sorted by name).
        let reordered = vec![
            t("a", Dtype::U8, &[2], vec![1, 2]),
            t("b", Dtype::U8, &[3], vec![9, 9, 9]),
        ];
        assert_eq!(serialize(&reordered, &meta).unwrap(), one);
    }

    #[test]
    fn empty_container_round_trips() {
        let bytes = serialize(&[], &BTreeMap::new()).unwrap();
        let view = parse(&bytes).unwrap();
        assert!(view.names().is_empty());
        assert!(view.metadata().is_empty());
    }

    #[test]
    fn rejects_too_short() {
        assert_eq!(parse(&[0u8; 4]), Err(HssError::TooShort));
        assert_eq!(parse(&[]), Err(HssError::TooShort));
    }

    #[test]
    fn rejects_header_larger_than_buffer() {
        // header_len = 1000 but no header bytes follow.
        let mut buf = Vec::new();
        buf.extend_from_slice(&1000u64.to_le_bytes());
        assert_eq!(parse(&buf), Err(HssError::TooShort));
    }

    #[test]
    fn rejects_oversized_header_len() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(MAX_HEADER_LEN + 1).to_le_bytes());
        assert_eq!(
            parse(&buf),
            Err(HssError::HeaderTooLarge(MAX_HEADER_LEN + 1))
        );
    }

    #[test]
    fn rejects_offset_past_data() {
        // Hand-craft a header whose tensor claims more data than exists.
        let header = r#"{"x":{"dtype":"U8","shape":[8],"data_offsets":[0,8]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&[0u8; 4]); // only 4 data bytes, tensor wants 8
        match parse(&buf) {
            Err(HssError::OffsetOutOfBounds { end, data_len, .. }) => {
                assert_eq!(end, 8);
                assert_eq!(data_len, 4);
            }
            other => panic!("expected OffsetOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn rejects_shape_length_mismatch() {
        // shape [4] of F32 => 16 bytes, but offsets only span 8.
        let header = r#"{"x":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        match parse(&buf) {
            Err(HssError::LengthMismatch { expected, got, .. }) => {
                assert_eq!(expected, 16);
                assert_eq!(got, 8);
            }
            other => panic!("expected LengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_unknown_dtype() {
        let header = r#"{"x":{"dtype":"F8_E4M3","shape":[1],"data_offsets":[0,1]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&[0u8; 1]);
        assert!(matches!(parse(&buf), Err(HssError::UnknownDtype(_))));
    }

    #[test]
    fn rejects_offset_order() {
        let header = r#"{"x":{"dtype":"U8","shape":[0],"data_offsets":[8,4]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        assert!(matches!(parse(&buf), Err(HssError::OffsetOrder { .. })));
    }

    #[test]
    fn rejects_invalid_json_header() {
        let header = b"{not json";
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header);
        assert!(matches!(parse(&buf), Err(HssError::HeaderJson(_))));
    }

    #[test]
    fn serialize_rejects_reserved_name() {
        let bad = vec![t("__metadata__", Dtype::U8, &[1], vec![0])];
        assert_eq!(
            serialize(&bad, &BTreeMap::new()),
            Err(HssError::ReservedName)
        );
    }

    /// Fuzz smoke: a deterministic pseudo-random stream of arbitrary buffers must
    /// never make `parse` panic — it may only return `Ok` or `Err`. This is the
    /// CI-friendly stand-in for the optional `cargo fuzz` target in `fuzz/`.
    #[test]
    fn parse_never_panics_on_random_input() {
        // Simple xorshift64* PRNG — no external crate, fully deterministic.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545F4914F6CDD1D)
        };
        for _ in 0..20_000 {
            let len = (next() % 256) as usize;
            let mut buf = Vec::with_capacity(len);
            for _ in 0..len {
                buf.push((next() & 0xFF) as u8);
            }
            // Must not panic; result is intentionally ignored.
            let _ = parse(&buf);
        }
    }

    /// Fuzz smoke, structured variant: take a valid container and corrupt the
    /// 8-byte length prefix to many values; parse must stay panic-free.
    #[test]
    fn corrupted_length_prefix_never_panics() {
        let tensors = vec![t("x", Dtype::F32, &[2, 2], vec![0u8; 16])];
        let mut bytes = serialize(&tensors, &BTreeMap::new()).unwrap();
        for hi in 0..=255u8 {
            bytes[7] = hi;
            for lo in (0..=255u8).step_by(17) {
                bytes[0] = lo;
                let _ = parse(&bytes);
            }
        }
    }
}
