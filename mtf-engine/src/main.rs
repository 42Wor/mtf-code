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
    /// Loads an MTF v2.0 file using memory mapping and processes trailing footer metadata [2]
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len();

        if file_size < 128 {
            return Err(MtfError::InvalidFormat(
                "File size too small to contain valid headers".into(),
            ));
        }

        let mmap = unsafe { Mmap::map(&file)? };

        // Step 1: Validate physical header magic sequence [2]
        if &mmap[0..8] != MAGIC_BYTES {
            return Err(MtfError::InvalidFormat(
                "Invalid MTF physical header magic".into(),
            ));
        }

        // Step 2: Validate trailing footer magic sequence [2]
        let footer_start = (file_size - 64) as usize;
        if &mmap[footer_start + 56..footer_start + 64] != MAGIC_FOOTER {
            return Err(MtfError::InvalidFormat(
                "Corrupt or missing trailing footer magic".into(),
            ));
        }

        // Step 3: Decode trailing segment pointers from footer [2]
        let index_offset =
            u64::from_le_bytes(mmap[footer_start..footer_start + 8].try_into().unwrap()) as usize;
        let metadata_offset = u64::from_le_bytes(
            mmap[footer_start + 8..footer_start + 16]
                .try_into()
                .unwrap(),
        ) as usize;
        let metadata_size = u64::from_le_bytes(
            mmap[footer_start + 16..footer_start + 24]
                .try_into()
                .unwrap(),
        ) as usize;

        // Step 4: Decompress and parse metadata payload (Zstandard Level 3) [2]
        let compressed_meta = &mmap[metadata_offset..metadata_offset + metadata_size];
        let mut decoder = zstd::Decoder::new(compressed_meta)?;
        let mut metadata_json = String::new();
        decoder.read_to_string(&mut metadata_json)?;

        // Step 5: Read structured tensor index segment [2]
        let mut tensors = Vec::new();
        let index_bytes = &mmap[index_offset..metadata_offset];
        let mut cursor = 0;

        while cursor < index_bytes.len() {
            if cursor + 18 > index_bytes.len() {
                break;
            }
            let name_hash = u64::from_le_bytes(index_bytes[cursor..cursor + 8].try_into().unwrap());
            let n_dims = index_bytes[cursor + 8] as usize;
            let quant_type = index_bytes[cursor + 9];
            let offset =
                u64::from_le_bytes(index_bytes[cursor + 10..cursor + 18].try_into().unwrap());
            cursor += 18;

            if cursor + (n_dims * 4) > index_bytes.len() {
                return Err(MtfError::InvalidFormat(
                    "Corrupted dimension layouts detected in index segment".into(),
                ));
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

    /// Performs an optimal O(log N) binary search lookup over pre-sorted hashes [2]
    pub fn get_tensor_payload(&self, name_hash: u64) -> Result<&[u8]> {
        match self
            .tensors
            .binary_search_by_key(&name_hash, |t| t.name_hash)
        {
            Ok(idx) => {
                let tensor = &self.tensors[idx];
                let start_offset = tensor.offset as usize;

                // Read exact tensor boundary by looking at next element or start of index
                let end_offset = if idx + 1 < self.tensors.len() {
                    self.tensors[idx + 1].offset as usize
                } else {
                    let footer_start = self.mmap.len() - 64;
                    let index_offset = u64::from_le_bytes(
                        self.mmap[footer_start..footer_start + 8]
                            .try_into()
                            .unwrap(),
                    ) as usize;
                    index_offset
                };

                Ok(&self.mmap[start_offset..end_offset])
            }
            Err(_) => Err(MtfError::TensorNotFound(format!(
                "Hash identifier {:x} is absent in target container.",
                name_hash
            ))),
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
    println!("[MTF Engine] Initializing systems-level core...");

    let model_path = "model.mtf";

    // Fallback: Inform if file needs compilation
    if !Path::new(model_path).exists() {
        println!(
            "[-] MTF model file '{}' not found. Please run 'cargo run --bin mtf-compiler' first.",
            model_path
        );
        return Ok(());
    }

    let model = MtfModel::load(model_path)?;
    println!("[+] Successfully parsed MTF v2.0 binary layout via memory mapping.");
    println!("[+] Metadata block retrieved:\n{}", model.get_metadata());

    // Display loaded tensors details and alignment verification
    println!("\n[+] Registered database tensors (Verified via FNV-1a binary search indexing):");
    for (i, t) in model.tensors().iter().enumerate() {
        let align_chk = if t.offset % 256 == 0 {
            "ALIGNED (256B)"
        } else {
            "MISALIGNED!"
        };
        println!(
            "  - Tensor #{}: Hash: {:x}, Offset: {} [{}], Shape: {:?}",
            i, t.name_hash, t.offset, align_chk, t.shape
        );
    }

    // Query active matrix elements
    let query_name = "model.layers.0.self_attn.q_proj.weight";
    let hash = mtf_hash_name(query_name);
    println!(
        "\n[*] Requesting tensor '{}' (Hash: {:x})...",
        query_name, hash
    );

    match model.get_tensor_payload(hash) {
        Ok(payload) => {
            println!(
                "[SUCCESS] Extracted payload ({} bytes total). First 4 floating-point elements:",
                payload.len()
            );
            // Convert first few bytes back to F32
            if payload.len() >= 16 {
                let floats: Vec<f32> = payload[0..16]
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                println!("  {:?}", floats);
            }
        }
        Err(e) => println!("[-] Error: {}", e),
    }

    Ok(())
}
