use anyhow::{Context, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{Device, Tensor};
use memmap2::Mmap;
use mtf_common::MAGIC_BYTES;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;

pub struct MtfEngine {
    mmap: Mmap,
    tensor_index: HashMap<u64, TensorEntry>,
    metadata: serde_json::Value,
    config: serde_json::Value,
}

struct TensorEntry {
    offset: u64,
    shape: Vec<usize>,
    dtype: u8, // 0 = F32
}

impl MtfEngine {
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = Cursor::new(&mmap[..]);
        let mut magic = [0u8; 8]; // Updated to match MAGIC_BYTES (8 bytes)
        cursor.read_exact(&mut magic)?;
        anyhow::ensure!(magic == *MAGIC_BYTES, "Invalid magic bytes");

        let _version = cursor.read_u32::<LittleEndian>()?;
        let tensor_count = cursor.read_u64::<LittleEndian>()?;
        cursor.seek(SeekFrom::Current(44))?;

        // Read footer from the end
        let file_len = mmap.len() as u64;
        let footer_start = file_len - 64; // Adjusted to match exact 64-byte footer format
        let mut footer_cursor = Cursor::new(&mmap[footer_start as usize..]);
        let index_offset = footer_cursor.read_u64::<LittleEndian>()?;
        let metadata_offset = footer_cursor.read_u64::<LittleEndian>()?;
        let metadata_size = footer_cursor.read_u64::<LittleEndian>()?;

        // Extract metadata block
        let meta_bytes =
            &mmap[metadata_offset as usize..metadata_offset as usize + metadata_size as usize];
        let decompressed = zstd::decode_all(meta_bytes)?;
        let meta_str = String::from_utf8(decompressed)?;
        let metadata: serde_json::Value = serde_json::from_str(&meta_str)?;
        let config = metadata
            .get("config")
            .cloned()
            .unwrap_or(serde_json::json!({}));

        // Parse tensor index
        let mut tensor_index = HashMap::new();
        let mut idx_cursor = Cursor::new(&mmap[index_offset as usize..]);
        for _ in 0..tensor_count {
            let hash = idx_cursor.read_u64::<LittleEndian>()?;
            let ndim = idx_cursor.read_u8()?;
            let dtype = idx_cursor.read_u8()?;
            let offset = idx_cursor.read_u64::<LittleEndian>()?;
            let mut shape = Vec::with_capacity(ndim as usize);
            for _ in 0..ndim {
                let dim = idx_cursor.read_u32::<LittleEndian>()? as usize;
                shape.push(dim);
            }
            tensor_index.insert(
                hash,
                TensorEntry {
                    offset,
                    shape,
                    dtype,
                },
            );
        }

        Ok(MtfEngine {
            mmap,
            tensor_index,
            metadata,
            config,
        })
    }

    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let hash = mtf_common::hash::mtf_hash_name(name);
        let entry = self
            .tensor_index
            .get(&hash)
            .with_context(|| format!("Tensor '{}' not found", name))?;
        let numel: usize = entry.shape.iter().product();
        let byte_offset = entry.offset as usize;
        let byte_len = numel * 4; // Assuming f32 (4 bytes)
        let data = &self.mmap[byte_offset..byte_offset + byte_len];

        // Convert raw bytes to f32 (little-endian)
        let float_data: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        // Updated to entry.shape.as_slice() to satisfy Shape trait boundary
        let tensor = Tensor::from_slice(&float_data, entry.shape.as_slice(), &Device::Cpu)?;
        Ok(tensor)
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        let hash = mtf_common::hash::mtf_hash_name(name);
        self.tensor_index.contains_key(&hash)
    }

    pub fn get_config(&self) -> &serde_json::Value {
        &self.config
    }
    pub fn get_metadata(&self) -> &serde_json::Value {
        &self.metadata
    }
}
