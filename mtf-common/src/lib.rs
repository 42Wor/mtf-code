pub mod hash;
pub mod format;

pub const MAGIC_BYTES: &[u8; 8] = b"MZTENSOR";
pub const FORMAT_VERSION: u32 = 1;
pub const MIN_ENGINE_VERSION: u32 = 1;
pub const ALIGNMENT_BOUNDARY: u64 = 256;
