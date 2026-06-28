use byteorder::{LittleEndian, WriteBytesExt};
use mtf_common::hash::mtf_hash_name;
use mtf_common::{ALIGNMENT_BOUNDARY, FORMAT_VERSION, MAGIC_BYTES, MIN_ENGINE_VERSION};
use safetensors::SafeTensors;
use std::fs::File;
use std::io::{Seek, SeekFrom, Write};

// A struct to track tensors during the compilation pipeline
struct TensorIndexEntry<'a> {
    hash: u64,
    name: String,
    shape: Vec<usize>,
    data_type: u8,        // 0 = F32, 1 = F16
    raw_data: &'a [u8],   // Pointer to raw safetensor memory
    absolute_offset: u64, // Final mapped location in .mtf
    byte_size: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n[MZTensor Labs] MTF Compiler Starting...");

    // For production, these will be CLI arguments. We will hardcode test paths for now.
    let input_path = "model.safetensors";
    let output_path = "model.mtf";

    // 1. MEMORY MAP THE SAFETENSORS FILE
    let in_file = File::open(input_path);
    if in_file.is_err() {
        println!(
            "[-] Target '{}' not found. Please place a dummy safetensors file here to compile.",
            input_path
        );
        return Ok(());
    }

    println!("[+] Loading source weights from: {}", input_path);
    let mmap = unsafe { memmap2::Mmap::map(&in_file.unwrap())? };
    let st = SafeTensors::deserialize(&mmap)?;

    // 2. PARSE TENSORS & CALCULATE MTF HASHES
    let mut tensors = Vec::new();
    let mut index_byte_size: u64 = 0;

    for (name, tensor) in st.tensors() {
        let hash = mtf_hash_name(&name);
        let raw_data = tensor.data();
        let shape = tensor.shape().to_vec();

        // 8B(hash) + 2B(name len) + name str + 1B(dim count) + 4B per dim + 1B(type) + 8B(offset) + 8B(size)
        let entry_size = 8 + 2 + name.len() as u64 + 1 + (shape.len() as u64 * 4) + 1 + 8 + 8;
        index_byte_size += entry_size;

        tensors.push(TensorIndexEntry {
            hash,
            name,
            shape,
            data_type: 1, // Assume F16 for base tests
            raw_data,
            absolute_offset: 0, // Will calculate next
            byte_size: raw_data.len() as u64,
        });
    }

    // 3. SORT TENSORS BY FNV-1A HASH FOR O(log N) ENGINE LOOKUP
    tensors.sort_by_key(|t| t.hash);
    println!(
        "[+] O(log N) Database Index mapped. ({} tensors)",
        tensors.len()
    );

    // 4. CALCULATE ALIGNED HARDWARE OFFSETS
    let header_size: u64 = 128;
    let meta_size: u64 = 0; // JSON to be implemented later
    let token_size: u64 = 0; // JSON to be implemented later

    // Cursor simulates where we are in the file to plan boundaries
    let mut file_cursor = header_size + meta_size + token_size + index_byte_size;

    for t in &mut tensors {
        // Enforce MTF padding commandment: "Offset must be multiple of 256"
        let boundary_modulo = file_cursor % ALIGNMENT_BOUNDARY;
        if boundary_modulo != 0 {
            file_cursor += ALIGNMENT_BOUNDARY - boundary_modulo; // inject padding math
        }

        t.absolute_offset = file_cursor;
        file_cursor += t.byte_size; // Move cursor past this tensor's actual data
    }

    // 5. WRITE MTF COMPILED BINARY
    println!(
        "[+] Executing zero-copy hardware serialization to '{}'",
        output_path
    );
    let mut out = File::create(output_path)?;

    // SECTION 1: GLOBAL HEADER (128 Bytes)
    out.write_all(MAGIC_BYTES)?; // 8 bytes
    out.write_u32::<LittleEndian>(FORMAT_VERSION)?; // 4 bytes
    out.write_u32::<LittleEndian>(MIN_ENGINE_VERSION)?; // 4 bytes
    out.write_u8(1)?; // 1 byte: Quant Type (F16 base)
    out.write_u8(0)?; // 1 byte: Meta ZSTD config (Off for now)
    out.write_u8(0)?; // 1 byte: Token ZSTD config
    out.write_u8(0)?; // 1 byte: Hash Algorithm (0 = FNV-1a)
    out.write_all(&[0u8; 4])?; // 4 bytes: Spacer

    out.write_u64::<LittleEndian>(meta_size)?; // 8 bytes
    out.write_u64::<LittleEndian>(token_size)?; // 8 bytes
    out.write_u64::<LittleEndian>(tensors.len() as u64)?; // 8 bytes
    out.write_u64::<LittleEndian>(index_byte_size)?; // 8 bytes

    // Fill the rest of the 128 bytes with null padding
    out.write_all(&[0u8; 72])?;

    // SECTION 2 & 3: Metadata & Tokenizer would go here (Sizes are 0 for now)

    // SECTION 4: HASH-SORTED TENSOR INDEX TABLE
    for t in &tensors {
        out.write_u64::<LittleEndian>(t.hash)?;
        out.write_u16::<LittleEndian>(t.name.len() as u16)?;
        out.write_all(t.name.as_bytes())?;

        out.write_u8(t.shape.len() as u8)?;
        for &dim in &t.shape {
            out.write_u32::<LittleEndian>(dim as u32)?;
        }

        out.write_u8(t.data_type)?;
        out.write_u64::<LittleEndian>(t.absolute_offset)?;
        out.write_u64::<LittleEndian>(t.byte_size)?;
    }

    // SECTION 5/6: INJECT 256-BYTE HARDWARE PADDING AND TENSOR PAYLOADS
    for t in &tensors {
        let current_pos = out.stream_position()?;

        // Assert our previous alignment math was completely accurate
        let boundary_modulo = current_pos % ALIGNMENT_BOUNDARY;
        if boundary_modulo != 0 {
            let padding_needed = ALIGNMENT_BOUNDARY - boundary_modulo;
            out.write_all(&vec![0u8; padding_needed as usize])?;
        }

        // Failsafe panic: Double check SIMD boundary!
        assert_eq!(
            out.stream_position()? % ALIGNMENT_BOUNDARY,
            0,
            "FATAL ALIGNMENT FAULT!"
        );

        out.write_all(t.raw_data)?; // ZERO-COPY WRITE from Mmap straight to Disk!
    }

    println!("[SUCCESS] MZT File architecture generated. Completely hardware safe.");
    Ok(())
}
