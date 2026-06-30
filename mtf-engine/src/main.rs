use memmap2::Mmap;
use mtf_common::hash::mtf_hash_name;
use mtf_common::{MAGIC_BYTES, MAGIC_FOOTER};
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug)]
pub enum MtfError {
    Io(std::io::Error),
    InvalidFormat(String),
    TensorNotFound(String),
}

impl std::error::Error for MtfError {}

impl fmt::Display for MtfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MtfError::Io(e) => write!(f, "I/O error: {}", e),
            MtfError::InvalidFormat(s) => write!(f, "Invalid MTF Format: {}", s),
            MtfError::TensorNotFound(s) => write!(f, "Tensor not found: {}", s),
        }
    }
}

impl From<std::io::Error> for MtfError {
    fn from(err: std::io::Error) -> Self {
        MtfError::Io(err)
    }
}

pub type Result<T> = std::result::Result<T, MtfError>;

#[derive(Debug, Clone)]
pub struct MtfTensorInfo {
    pub name_hash: u64,
    pub shape: Vec<u32>,
    pub offset: u64,
    pub quant_type: u8,
}

pub struct MtfModel {
    mmap: Mmap,
    tensors: Vec<MtfTensorInfo>,
    metadata_json: String,
}

impl MtfModel {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len();
        
        if file_size < 128 {
            return Err(MtfError::InvalidFormat("File size too small for standard MTF headers".into()));
        }

        let mmap = unsafe { Mmap::map(&file)? };

        // 1. Verify Header
        if &mmap[0..8] != MAGIC_BYTES {
            return Err(MtfError::InvalidFormat("Corrupt or invalid physical header magic".into()));
        }

        // 2. Verify Footer
        let footer_start = (file_size - 64) as usize;
        if &mmap[footer_start + 56..footer_start + 64] != MAGIC_FOOTER {
            return Err(MtfError::InvalidFormat("Footer structure magic validation failed".into()));
        }

        // 3. Decode Trailing Offsets
        let index_offset = u64::from_le_bytes(mmap[footer_start..footer_start + 8].try_into().unwrap()) as usize;
        let metadata_offset = u64::from_le_bytes(mmap[footer_start + 8..footer_start + 16].try_into().unwrap()) as usize;
        let metadata_size = u64::from_le_bytes(mmap[footer_start + 16..footer_start + 24].try_into().unwrap()) as usize;

        // 4. Decompress Metadata Sandbox
        let compressed_meta = &mmap[metadata_offset..metadata_offset + metadata_size];
        let mut decoder = zstd::Decoder::new(compressed_meta)?;
        let mut metadata_json = String::new();
        decoder.read_to_string(&mut metadata_json)?;

        // 5. Decode Index Segment
        let mut tensors = Vec::new();
        let index_bytes = &mmap[index_offset..metadata_offset];
        let mut cursor = 0;

        while cursor < index_bytes.len() {
            if cursor + 18 > index_bytes.len() { break; }
            let name_hash = u64::from_le_bytes(index_bytes[cursor..cursor + 8].try_into().unwrap());
            let n_dims = index_bytes[cursor + 8] as usize;
            let quant_type = index_bytes[cursor + 9];
            let offset = u64::from_le_bytes(index_bytes[cursor + 10..cursor + 18].try_into().unwrap());
            cursor += 18;

            if cursor + (n_dims * 4) > index_bytes.len() {
                return Err(MtfError::InvalidFormat("Corrupted dimension metadata in index segment".into()));
            }

            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let dim = u32::from_le_bytes(index_bytes[cursor..cursor + 4].try_into().unwrap());
                shape.push(dim);
                cursor += 4;
            }

            tensors.push(MtfTensorInfo {
                name_hash,
                shape,
                offset,
                quant_type,
            });
        }

        Ok(MtfModel {
            mmap,
            tensors,
            metadata_json,
        })
    }

    pub fn get_tensor_payload(&self, name_hash: u64) -> Result<&[u8]> {
        match self.tensors.binary_search_by_key(&name_hash, |t| t.name_hash) {
            Ok(idx) => {
                let tensor = &self.tensors[idx];
                let start_offset = tensor.offset as usize;
                
                let end_offset = if idx + 1 < self.tensors.len() {
                    self.tensors[idx + 1].offset as usize
                } else {
                    let footer_start = self.mmap.len() - 64;
                    let index_offset = u64::from_le_bytes(self.mmap[footer_start..footer_start + 8].try_into().unwrap()) as usize;
                    index_offset
                };

                Ok(&self.mmap[start_offset..end_offset])
            }
            Err(_) => Err(MtfError::TensorNotFound(format!("Hash key {:x} not present in binary index", name_hash))),
        }
    }

    pub fn get_metadata(&self) -> &str {
        &self.metadata_json
    }

    pub fn tensors(&self) -> &[MtfTensorInfo] {
        &self.tensors
    }
}

fn main() -> Result<()> {
    println!("\n[MTF Engine] Bootstrapping v2.0 Execution Core...");

    let model_path = "model.mtf";
    if !Path::new(model_path).exists() {
        println!("[-] Error: Compile target 'model.mtf' is missing.");
        std::process::exit(1);
    }

    let model = MtfModel::load(model_path)?;
    println!("[+] Load successful! Memory-mapped model into memory.");

    // Parse metadata sandbox as JSON object to pretty-print
    let parsed_meta: serde_json::Value = serde_json::from_str(model.get_metadata())
        .map_err(|e| MtfError::InvalidFormat(format!("Metadata is not valid JSON: {}", e)))?;
    println!("[+] Decompressed metadata block: \n{}", serde_json::to_string_pretty(&parsed_meta).unwrap());

    // Run alignment checks
    println!("\n[*] Performing strict SIMD alignment audit...");
    let mut align_errors = 0;
    for t in model.tensors() {
        if t.offset % 256 != 0 {
            align_errors += 1;
        }
    }
    if align_errors == 0 {
        println!("[+] Audit Passed: 100% of the {} tensors conform to the 256-byte boundary standard.", model.tensors().len());
    } else {
        println!("[-] Audit Failed: Found {} unaligned tensors.", align_errors);
    }

    // Try verifying some key Qwen2 weights
    let test_layers = vec![
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ];

    println!("\n[*] Running query lookups on critical transformer parameters:");
    for layer in test_layers {
        let hash = mtf_hash_name(layer);
        match model.get_tensor_payload(hash) {
            Ok(payload) => {
                println!("  [✓] Found '{}' (Hash: {:x}) - Size: {} bytes", layer, hash, payload.len());
            }
            Err(_) => {
                println!("  [✗] Missing '{}' (Hash: {:x})", layer, hash);
            }
        }
    }

    println!("\n[SUCCESS] Engine validation and runtime verification pipeline concluded.");
    Ok(())
}
