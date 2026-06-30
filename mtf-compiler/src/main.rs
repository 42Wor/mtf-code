use byteorder::{LittleEndian, WriteBytesExt};
use mtf_common::hash::mtf_hash_name;
use mtf_common::{
    ALIGNMENT_BOUNDARY, FORMAT_VERSION, MAGIC_BYTES, MAGIC_FOOTER, MIN_ENGINE_VERSION,
};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

struct CompileTensor<'a> {
    hash: u64,
    shape: Vec<usize>,
    data_type: u8,
    raw_data: &'a [u8],
    absolute_offset: u64,
}

fn create_dummy_safetensors_if_missing<P: AsRef<Path>>(
    path: P,
) -> Result<(), Box<dyn std::error::Error>> {
    if path.as_ref().exists() {
        return Ok(());
    }

    println!(
        "[*] Input '{}' not found. Creating a highly aligned mock SafeTensors file...",
        path.as_ref().display()
    );

    // Create actual raw bytes representing some weights
    let t1_data = vec![0.5f32; 512]; // F32
    let t2_data = vec![0.25f32; 1024]; // F32

    let t1_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(t1_data.as_ptr() as *const u8, t1_data.len() * 4) };
    let t2_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(t2_data.as_ptr() as *const u8, t2_data.len() * 4) };

    let mut tensors = HashMap::new();
    tensors.insert(
        "model.embed_tokens.weight".to_string(),
        TensorView::new(Dtype::F32, vec![16, 32], t1_bytes)?,
    );
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        TensorView::new(Dtype::F32, vec![32, 32], t2_bytes)?,
    );

    let serialized = safetensors::serialize(&tensors, &None)?;
    std::fs::write(path, serialized)?;
    println!("[+] Mock SafeTensors file created successfully.");
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n[MZTensor Labs] MTF v2.0 Compiler Bootstrapping...");

    let input_path = "model.safetensors";
    let output_path = "model.mtf";

    create_dummy_safetensors_if_missing(input_path)?;

    // 1. Memory-map the safetensors file
    let in_file = File::open(input_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&in_file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let mut tensors = Vec::new();

    // 2. Parse targets
    for (name, tensor) in st.tensors() {
        let hash = mtf_hash_name(&name);
        let raw_data = tensor.data();
        let shape = tensor.shape().to_vec();

        tensors.push(CompileTensor {
            hash,
            shape,
            data_type: 0, // F32 mapped as 0
            raw_data,
            absolute_offset: 0,
        });
    }

    // 3. Binary Search Optimization: Sort indices natively by FNV-1a Hash [1, 2]
    tensors.sort_by_key(|t| t.hash);
    println!("[+] Sorted {} tensors by FNV-1a hash key.", tensors.len());

    // 4. Begin single-pass compilation streaming [2]
    let mut out = File::create(output_path)?;

    // Step A: Write fixed 64-byte Header [2]
    out.write_all(MAGIC_BYTES)?; // 0..8
    out.write_u32::<LittleEndian>(FORMAT_VERSION)?; // 8..12
    out.write_u32::<LittleEndian>(MIN_ENGINE_VERSION)?; // 12..16
    out.write_u64::<LittleEndian>(tensors.len() as u64)?; // 16..24
    out.write_all(&[0u8; 40])?; // 24..64 (System Padding / Reserved)

    // Step B: Write aligned tensor payloads
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

    // Step C: Record Index segment offset and serialize index
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

    // Step D: Record metadata offset, serialize & compress sandbox payload (Zstandard Level 3) [2]
    let metadata_offset = out.stream_position()?;
    let mock_metadata = json!({
        "model_type": "maaz-transformer",
        "vocab_size": 32000,
        "hidden_size": 2048,
        "intermediate_size": 5632,
        "num_attention_heads": 32,
        "num_hidden_layers": 24,
        "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}"
    });
    let metadata_str = serde_json::to_string(&mock_metadata)?;
    let compressed_meta = zstd::encode_all(metadata_str.as_bytes(), 3)?;
    let metadata_size = compressed_meta.len() as u64;
    out.write_all(&compressed_meta)?;

    // Step E: Write trailing 64-byte Footer [2]
    out.write_u64::<LittleEndian>(index_offset)?;
    out.write_u64::<LittleEndian>(metadata_offset)?;
    out.write_u64::<LittleEndian>(metadata_size)?;
    out.write_all(&[0u8; 32])?; // System Padding / Reserved
    out.write_all(MAGIC_FOOTER)?; // Footer Magic 'MZTFOOTR'

    println!(
        "[SUCCESS] MTF v2.0 binary successfully compiled to '{}'.",
        output_path
    );
    Ok(())
}
