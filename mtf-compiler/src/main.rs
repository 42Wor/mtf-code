use byteorder::{LittleEndian, WriteBytesExt};
use mtf_common::hash::mtf_hash_name;
use mtf_common::{ALIGNMENT_BOUNDARY, FORMAT_VERSION, MAGIC_BYTES, MAGIC_FOOTER, MIN_ENGINE_VERSION};
use safetensors::SafeTensors;
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::Path;

struct CompileTensor<'a> {
    hash: u64,
    shape: Vec<usize>,
    data_type: u8,
    raw_data: &'a [u8],
    absolute_offset: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n[MZTensor Labs] Starting MTF v2.0 Compiler Pipeline...");

    let possible_paths = vec![
        "test_models/qwen2-0.5b/model.safetensors",
        "/mnt/shared/mallow/mtf-code/test_models/qwen2-0.5b/model.safetensors",
        "model.safetensors"
    ];

    let mut input_path = "";
    for path in possible_paths {
        if Path::new(path).exists() {
            input_path = path;
            break;
        }
    }

    if input_path.is_empty() {
        println!("[-] Error: No source safetensors file found. Please ensure model.safetensors exists.");
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

        let data_type = match tensor.dtype() {
            safetensors::Dtype::F32 => 0,
            safetensors::Dtype::F16 => 1,
            safetensors::Dtype::BF16 => 1, 
            _ => 1,
        };

        tensors.push(CompileTensor {
            hash,
            shape,
            data_type,
            raw_data,
            absolute_offset: 0,
        });
    }

    // Sort by hash key for O(log N) binary search lookup
    tensors.sort_by_key(|t| t.hash);
    println!("[+] Index built & sorted. Processing {} tensors...", tensors.len());

    let output_path = "model.mtf";
    let mut out = File::create(output_path)?;

    // 1. Write Header (64 bytes)
    out.write_all(MAGIC_BYTES)?; 
    out.write_u32::<LittleEndian>(FORMAT_VERSION)?; 
    out.write_u32::<LittleEndian>(MIN_ENGINE_VERSION)?; 
    out.write_u64::<LittleEndian>(tensors.len() as u64)?; 
    out.write_all(&[0u8; 40])?; 

    // 2. Write aligned payloads
    for t in &mut tensors {
        let current_pos = out.stream_position()?;
        let boundary_modulo = current_pos % ALIGNMENT_BOUNDARY;
        if boundary_modulo != 0 {
            let padding_needed = ALIGNMENT_BOUNDARY - boundary_modulo;
            out.write_all(&vec![0u8; padding_needed as usize])?;
        }

        t.absolute_offset = out.stream_position()?;
        out.write_all(t.raw_data)?;
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

    println!("[SUCCESS] MTF v2.0 compilation finished successfully. Output: {}", output_path);
    Ok(())
}
