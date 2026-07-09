use crate::metadata::build_metadata_json;
use crate::tensor::{process_tensor_data, CompileTensor};
use crate::validator::validate_tensor_shapes;
use anyhow::{Context, Result};
use byteorder::{LittleEndian, WriteBytesExt};
use indicatif::{ProgressBar, ProgressStyle};
use mtf_common::hash::mtf_hash_name;
use mtf_common::{ALIGNMENT_BOUNDARY, FORMAT_VERSION, MAGIC_BYTES, MAGIC_FOOTER};
use safetensors::SafeTensors;
use std::fs::File;
use std::io::{Seek, Write};
use std::path::Path;



pub fn run_compile(input_dir: &Path, output_path: &Path) -> Result<()> {
    let safetensors_path = input_dir.join("model.safetensors");
    if !safetensors_path.exists() {
        anyhow::bail!("model.safetensors not found in {:?}", input_dir);
    }

    log::info!("Assembling metadata bundle...");
    let metadata_str = build_metadata_json(input_dir)?;
    let compressed_meta =
        zstd::encode_all(metadata_str.as_bytes(), 3).context("Failed to compress metadata")?;
    let metadata_size = compressed_meta.len() as u64;

    let config_path = input_dir.join("config.json");
    let config_json: Option<serde_json::Value> = if config_path.exists() {
        let config_str = std::fs::read_to_string(&config_path)?;
        Some(serde_json::from_str(&config_str)?)
    } else {
        log::warn!("config.json not found – skipping shape validation");
        None
    };

    log::info!("Memory mapping safetensors...");
    let in_file = File::open(&safetensors_path)?;
    let mmap = unsafe { memmap2::Mmap::map(&in_file)? };
    let st = SafeTensors::deserialize(&mmap)?;

    let tensor_count = st.tensors().len();
    let pb = ProgressBar::new(tensor_count as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message("Converting tensors");

    let mut tensors = Vec::with_capacity(tensor_count);
    let mut tensor_names = Vec::new();

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
        log::debug!(
            "Tensor: {} ({} -> F32) shape = {:?}",
            name,
            source_type_str,
            shape
        );

        tensors.push(CompileTensor {
            hash,
            shape: shape.clone(),
            data_type: 0,
            raw_data: converted_data,
            absolute_offset: 0,
        });
        tensor_names.push((name.clone(), shape));
        pb.inc(1);
    }

    pb.finish_with_message("Tensor conversion complete");

    if let Some(config) = config_json {
        if let Err(e) = validate_tensor_shapes(&tensor_names, &config) {
            log::warn!("Shape validation issues: {}", e);
        } else {
            log::info!("All tensor shapes validated against config.");
        }
    }

    tensors.sort_by_key(|t| t.hash);
    log::info!(
        "Index built and sorted. Writing {} tensors...",
        tensors.len()
    );

    let mut out = File::create(output_path)?;
    let write_pb = ProgressBar::new(tensors.len() as u64);
    write_pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} (writing)",
            )
            .unwrap()
            .progress_chars("#>-"),
    );

    // Header
    out.write_all(MAGIC_BYTES)?;
    out.write_u32::<LittleEndian>(FORMAT_VERSION)?;
    out.write_u64::<LittleEndian>(tensors.len() as u64)?;
    out.write_all(&[0u8; 44])?;

    // Payloads
    for t in &mut tensors {
        let current_pos = out.stream_position()?;
        let boundary_modulo = current_pos % ALIGNMENT_BOUNDARY;
        if boundary_modulo != 0 {
            let padding_needed = ALIGNMENT_BOUNDARY - boundary_modulo;
            out.write_all(&vec![0u8; padding_needed as usize])?;
        }
        t.absolute_offset = out.stream_position()?;
        out.write_all(t.raw_data.as_slice())?;
        write_pb.inc(1);
    }
    write_pb.finish_with_message("Payload written");

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

    // Metadata
    let metadata_offset = out.stream_position()?;
    out.write_all(&compressed_meta)?;

    // Footer
    out.write_u64::<LittleEndian>(index_offset)?;
    out.write_u64::<LittleEndian>(metadata_offset)?;
    out.write_u64::<LittleEndian>(metadata_size)?;
    out.write_u32::<LittleEndian>(FORMAT_VERSION)?;
    out.write_all(&[0u8; 28])?;
    out.write_all(MAGIC_FOOTER)?;

    log::info!("MTF file written to {:?}", output_path);
    Ok(())
}
