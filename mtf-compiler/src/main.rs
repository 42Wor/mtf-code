use byteorder::{LittleEndian, WriteBytesExt};
use mtf_common::hash::mtf_hash_name;
use mtf_common::{ALIGNMENT_BOUNDARY, MAGIC_BYTES, MAGIC_FOOTER};
use safetensors::SafeTensors;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::Path;

/// Simple enum to manage either zero-copy borrowed slices or allocated transformed data buffers.
enum TensorData<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl<'a> TensorData<'a> {
    fn as_slice(&self) -> &[u8] {
        match self {
            TensorData::Borrowed(s) => s,
            TensorData::Owned(v) => v,
        }
    }
}

/// Helper structure for tracking and processing compilation payloads
struct CompileTensor<'a> {
    hash: u64,
    #[allow(dead_code)]
    name: String,
    shape: Vec<usize>,
    data_type: u8,
    raw_data: TensorData<'a>,
    absolute_offset: u64,
}

/// IEEE-754 Single-Precision (32-bit) to Half-Precision (16-bit) Converter.
fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127;
    let mut mant = bits & 0x7fffff;

    if exp == 128 {
        let m = if mant != 0 { 0x200 } else { 0 };
        return (sign | 0x7c00 | m) as u16;
    }

    exp += 15; // bias to half

    if exp >= 31 {
        // Overflow to infinity
        return (sign | 0x7c00) as u16;
    }

    if exp <= 0 {
        // Underflow
        if exp < -10 {
            return sign as u16;
        }
        // Denormalize
        mant = (mant | 0x800000) >> (1 - exp);
        return (sign | (mant >> 13)) as u16;
    }

    // Normal half representation
    (sign | ((exp as u32) << 10) | (mant >> 13)) as u16
}

/// Decodes an FP16 / BF16 byte layout into standard F32 format.
fn decode_half(bytes: [u8; 2], is_bf16: bool) -> f32 {
    if is_bf16 {
        let u = u16::from_le_bytes(bytes);
        f32::from_bits((u as u32) << 16)
    } else {
        let h = u16::from_le_bytes(bytes);
        let sign = (h >> 15) & 1;
        let exp = (h >> 10) & 0x1f;
        let mant = h & 0x3ff;
        if exp == 0 {
            let sign_f = if sign == 1 { -1.0 } else { 1.0 };
            sign_f * (mant as f32) * 2.0f32.powi(-24)
        } else if exp == 31 {
            if mant == 0 {
                if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            }
        } else {
            let sign_f = if sign == 1 { -1.0 } else { 1.0 };
            sign_f * (1.0 + (mant as f32) / 1024.0) * 2.0f32.powi(exp as i32 - 15)
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n[MZTensor Labs] Starting MTF Compiler Pipeline...");

    let possible_paths = vec![
        "test_models/qwen2-0.5b/model.safetensors",
        "/mnt/shared/mallow/mtf-code/test_models/qwen2-0.5b/model.safetensors",
        "model.safetensors",
    ];

    let mut input_path = "";
    for path in possible_paths {
        if Path::new(path).exists() {
            input_path = path;
            break;
        }
    }

    if input_path.is_empty() {
        println!(
            "[-] Error: No source safetensors file found. Please ensure model.safetensors exists."
        );
        std::process::exit(1);
    }

    println!("[+] Found input safetensors model at: {}", input_path);

    let parent_dir = Path::new(input_path).parent().unwrap();
    let config_path = parent_dir.join("config.json");

    // Load SafeTensors
    let in_file = File::open(input_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&in_file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mut tensors = Vec::new();
    for (name, tensor) in st.tensors() {
        let hash = mtf_hash_name(&name);
        let raw_data = tensor.data();
        let shape = tensor.shape().to_vec();

        let source_type_str = match tensor.dtype() {
            safetensors::Dtype::F32 => "F32",
            safetensors::Dtype::F16 => "F16",
            safetensors::Dtype::BF16 => "BF16",
            _ => "Unknown",
        };

        // Rule: Norms stay F32 (0) for high precision, other weights compress to F16 (1).
        let (target_type_str, data_type) = if name.contains("norm") {
            ("F32", 0)
        } else {
            ("F16", 1)
        };

        // Convert the data if needed
        let converted_data = match (tensor.dtype(), data_type) {
            (safetensors::Dtype::F32, 1) => {
                // Convert F32 -> F16
                let elements = raw_data.len() / 4;
                let mut converted = Vec::with_capacity(elements * 2);
                for chunk in raw_data.chunks_exact(4) {
                    let val = f32::from_le_bytes(chunk.try_into().unwrap());
                    let h = f32_to_f16(val);
                    converted.extend_from_slice(&h.to_le_bytes());
                }
                TensorData::Owned(converted)
            }
            (safetensors::Dtype::F16, 0) | (safetensors::Dtype::BF16, 0) => {
                // Convert F16/BF16 -> F32
                let elements = raw_data.len() / 2;
                let mut converted = Vec::with_capacity(elements * 4);
                let is_bf16 = tensor.dtype() == safetensors::Dtype::BF16;
                for chunk in raw_data.chunks_exact(2) {
                    let val = decode_half(chunk.try_into().unwrap(), is_bf16);
                    converted.extend_from_slice(&val.to_le_bytes());
                }
                TensorData::Owned(converted)
            }
            _ => {
                // Keep as-is
                TensorData::Borrowed(raw_data)
            }
        };

        println!(
            "INFO:mtf-compiler: {:<40} {:>5} --> {:<4}, shape = {:?}",
            name, source_type_str, target_type_str, shape
        );

        tensors.push(CompileTensor {
            hash,
            name: name.clone(),
            shape,
            data_type,
            raw_data: converted_data,
            absolute_offset: 0,
        });
    }

    // Sort by hash key for O(log N) binary search lookup
    tensors.sort_by_key(|t| t.hash);
    println!(
        "[+] Index built & sorted. Processing {} tensors...",
        tensors.len()
    );

    let output_path = "model.mtf";
    let mut out = File::create(output_path)?;

    // 1. Write Header (64 bytes) - Versioning removed, replaced with padding
    out.write_all(MAGIC_BYTES)?;
    out.write_u64::<LittleEndian>(tensors.len() as u64)?;
    out.write_all(&[0u8; 48])?; // 8 + 8 + 48 = 64 bytes

    // 2. Write aligned payloads
    for t in &mut tensors {
        let current_pos = out.stream_position()?;
        let boundary_modulo = current_pos % ALIGNMENT_BOUNDARY;
        if boundary_modulo != 0 {
            let padding_needed = ALIGNMENT_BOUNDARY - boundary_modulo;
            out.write_all(&vec![0u8; padding_needed as usize])?;
        }

        t.absolute_offset = out.stream_position()?;
        out.write_all(t.raw_data.as_slice())?;
    }

    // 3. Write Index segment
    let index_offset = out.stream_position()?;
    for t in &tensors {
        out.write_u64::<LittleEndian>(t.hash)?;
        out.write_u8(t.shape.len() as u8)?;
        out.write_u8(t.data_type)?;
        out.write_u64::<LittleEndian>(t.absolute_offset)?;
        for &dim in &t.shape {
            out.write_u32::<LittleEndian>(dim as u32)?;
        }
    }

    // 4. Read config or use fallback JSON, compress with Zstd Level 3
    let mut metadata_str = String::new();
    if config_path.exists() {
        println!("[+] Found config.json at: {}", config_path.display());
        let mut file = File::open(config_path)?;
        file.read_to_string(&mut metadata_str)?;
    } else {
        println!("[-] No config.json found. Creating generic fallback metadata.");
        metadata_str = r#"{"model_type": "generic-transformer", "vocab_size": 32000}"#.to_string();
    }

    let metadata_offset = out.stream_position()?;
    let compressed_meta = zstd::encode_all(metadata_str.as_bytes(), 3)?;
    let metadata_size = compressed_meta.len() as u64;
    out.write_all(&compressed_meta)?;

    // 5. Write Footer (64 bytes)
    out.write_u64::<LittleEndian>(index_offset)?;
    out.write_u64::<LittleEndian>(metadata_offset)?;
    out.write_u64::<LittleEndian>(metadata_size)?;
    out.write_all(&[0u8; 32])?;
    out.write_all(MAGIC_FOOTER)?;

    println!(
        "[SUCCESS] MTF compilation finished successfully. Output: {}",
        output_path
    );
    Ok(())
}
