pub mod hash;
pub mod format;

pub const MAGIC_BYTES: &[u8; 8] = b"MZTENSOR";
pub const MAGIC_FOOTER: &[u8; 8] = b"MZTFOOTR";
pub const FORMAT_VERSION: u32 = 2; 
pub const MIN_ENGINE_VERSION: u32 = 2;
pub const ALIGNMENT_BOUNDARY: u64 = 256;
