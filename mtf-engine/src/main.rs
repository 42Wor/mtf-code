use memmap2::Mmap;
use mtf_common::{ALIGNMENT_BOUNDARY, MAGIC_BYTES};
use std::fs::File;
use std::io::{Read, Result};

fn load_mtf(path: &str) -> Result<()> {
    let file = File::open(path)?;
    // Zero-Copy Page-Fault Mapping directly through the OS!
    let mmap = unsafe { Mmap::map(&file)? };

    if mmap.len() < 128 {
        panic!("File shorter than strict 128-byte MTF Global Header length.");
    }

    let magic = &mmap[0..8];
    if magic != MAGIC_BYTES {
        panic!("Fatal: Corrupt Header. Expected MZTENSOR magic string.");
    }

    println!("[MTF Engine] Boot successful.");
    println!(
        "[MTF Engine] Loaded {} via pure zero-copy memory mapping.",
        path
    );
    Ok(())
}

fn main() -> Result<()> {
    println!("[MTF Engine] v1.0 standing by for inference.");

    // Attempt to load the MTF file created by the compiler
    if let Err(e) = load_mtf("../output_model.mtf") {
        println!("[MTF Engine] Could not load model: {}", e);
    }

    Ok(())
}
