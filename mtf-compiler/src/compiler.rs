use crate::metadata::build_metadata_json;
use crate::tensor::{process_tensor_data, CompileTensor};
use byteorder::{LittleEndian, WriteBytesExt};
use mtf_common::hash::mtf_hash_name;
use mtf_common::{ALIGNMENT_BOUNDARY, MAGIC_BYTES, MAGIC_FOOTER};
use safetensors::SafeTensors;
use std::fs::File;
use std::io::{Seek, Write};
use std::path::Path;

pub fn run_compile(input_dir: &Path, output_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Find safetensors file
    let safetensors_path = input_dir.join("model.safetensors");
    if !safetensors_path.exists() {
        return Err(format!("model.safetensors not found in {:?}", input_dir).into());
    }

    // 2. Build Metadata (Config + Tokenizer)
    println!("[*] Assembling metadata bundle...");
    let metadata_str = build_metadata_json(input_dir)?;
    let compressed_meta = zstd::encode_all(metadata_str.as_bytes(), 3)?;
    let metadata_size = compressed_meta.len() as u64;

    // 3. Process Tensors
    println!("[*] Memory mapping safetensors...");
    let in_file = File::open(&safetensors_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&in_file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mut tensors = Vec::new();
    
    println!("[*] Processing tensors (Multi-threaded F16/BF16 -> F32 conversion)...");
    for (name, tensor) in st.tensors() {
        let hash = mtf_hash_name(&name);
        let raw_data = tensor.data();
        let shape = tensor.shape().to_vec();
        let dtype = tensor.dtype();

        let source_type_str = match dtype {
            safetensors::Dtype::F32 => "F32",
            safetensors::Dtype::F16 => "F16",
            safetensors::Dtype::BF16 => "BF16",
            _ => "Unknown",
        };

        let converted_data = process_tensor_data(raw_data, dtype);

        println!(
            "  -> {:<40} {:>5} --> F32, shape = {:?}",
            name, source_type_str, shape
        );

        tensors.push(CompileTensor {
            hash,
            name: name.clone(),
            shape,
            data_type: 0, // 0 = F32
            raw_data: converted_data,
            absolute_offset: 0,
        });
    }

    tensors.sort_by_key(|t| t.hash);
    println!("[+] Index built & sorted. Writing {} tensors to disk...", tensors.len());

    // 4. Write MTF File
    let mut out = File::create(output_path)?;

    // Header
    out.write_all(MAGIC_BYTES)?;
    out.write_u64::<LittleEndian>(tensors.len() as u64)?;
    out.write_all(&[0u8; 48])?;

    // Tensor Payloads (Aligned)
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

    // Index
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

    // Metadata Block
    let metadata_offset = out.stream_position()?;
    out.write_all(&compressed_meta)?;

    // Footer
    out.write_u64::<LittleEndian>(index_offset)?;
    out.write_u64::<LittleEndian>(metadata_offset)?;
    out.write_u64::<LittleEndian>(metadata_size)?;
    out.write_all(&[0u8; 32])?;
    out.write_all(MAGIC_FOOTER)?;

    println!("[SUCCESS] MTF compilation finished successfully. Output: {:?}", output_path);
    Ok(())
}
