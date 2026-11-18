//! Minimal mmap'd GGUF v3 reader — kennel's window into containers.
//! This is the rs roadmap's "gguf pass1" landing as an engine asset.
//!
//! Handles: header, full metadata KV table, tensor infos, data section
//! offsets. Type table includes the fork's K9 types (43–48), whose
//! superblock layout is QK9_SUPER=256 / QK9_GROUP=64 (ggml-common.h).

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context, Result};

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" LE

/// fork ggml.h type values kennel knows by name (43–48 = K9 family)
pub fn type_name(ty: u32) -> String {
    let s = match ty {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "Q8_K",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        30 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "IQ4_NL",
        43 => "K9 (rANS stream sentinel)",
        44 => "K9_4",
        45 => "K9_6",
        46 => "K9_7",
        47 => "K9_4KV",
        48 => "K9_6KV",
        _ => return format!("type {ty}"),
    };
    s.to_string()
}

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    /// short single-line rendering for `kennel info`
    pub fn brief(&self) -> String {
        match self {
            Value::Str(s) => format!("{s:?}"),
            Value::Array(v) => {
                if v.len() <= 8 {
                    let inner: Vec<String> = v.iter().map(|x| x.brief()).collect();
                    format!("[{}]", inner.join(", "))
                } else {
                    format!("[{} elems: {}, …]", v.len(), v[0].brief())
                }
            }
            Value::F32(x) => format!("{x}"),
            Value::F64(x) => format!("{x}"),
            other => format!("{other:?}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>, // GGUF order: fastest-varying first
    pub ty: u32,
    pub offset: u64, // relative to data section start
}

pub struct Gguf {
    pub version: u32,
    pub kv: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    pub data_offset: u64,
    pub mmap: memmap2::Mmap,
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.b.len() {
            bail!("gguf truncated at {} (+{n})", self.pos);
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        if n > 1 << 20 {
            bail!("implausible string length {n}");
        }
        let s = self.take(n)?;
        Ok(String::from_utf8_lossy(s).into_owned())
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U8(self.u8()?),
            1 => Value::I8(self.u8()? as i8),
            2 => Value::U16(self.u16()?),
            3 => Value::I16(self.u16()? as i16),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(self.f32()?),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let elem = self.u32()?;
                let n = self.u64()? as usize;
                if n > 1 << 24 {
                    bail!("implausible array length {n}");
                }
                let mut v = Vec::with_capacity(n.min(4096));
                for _ in 0..n {
                    v.push(self.value(elem)?);
                }
                Value::Array(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(self.f64()?),
            other => bail!("unknown gguf value type {other}"),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| path.display().to_string())?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let mut c = Cursor { b: &mmap, pos: 0 };
        let magic = c.u32()?;
        if magic != GGUF_MAGIC {
            bail!("not a GGUF file (magic {magic:#x})");
        }
        let version = c.u32()?;
        if version != 3 {
            bail!("gguf version {version}, kennel reads v3");
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;

        let mut kv = BTreeMap::new();
        for _ in 0..n_kv {
            let key = c.string()?;
            let ty = c.u32()?;
            let val = c.value(ty)?;
            kv.insert(key, val);
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let n_dims = c.u32()? as usize;
            if n_dims > 8 {
                bail!("tensor {name}: implausible n_dims {n_dims}");
            }
            let mut dims = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                dims.push(c.u64()?);
            }
            let ty = c.u32()?;
            let offset = c.u64()?;
            tensors.push(TensorInfo { name, dims, ty, offset });
        }

        let alignment = match kv.get("general.alignment") {
            Some(Value::U32(a)) => *a as u64,
            _ => 32,
        };
        let data_offset = (c.pos as u64).div_ceil(alignment) * alignment;

        Ok(Self { version, kv, tensors, alignment, data_offset, mmap })
    }

    /// absolute byte range of a tensor's blob within the mmap
    pub fn tensor_bytes(&self, t: &TensorInfo) -> Result<&[u8]> {
        let start = self.data_offset + t.offset;
        // end: next tensor's start, else EOF — GGUF blobs are tightly packed
        let end = self
            .tensors
            .iter()
            .map(|x| self.data_offset + x.offset)
            .filter(|&o| o > start)
            .min()
            .unwrap_or(self.mmap.len() as u64) as usize;
        let start = start as usize;
        if start > self.mmap.len() || end > self.mmap.len() {
            bail!("tensor {} out of file range", t.name);
        }
        Ok(&self.mmap[start..end])
    }
}
