/// The ultra-fast FNV-1a (64-bit) algorithm replacing slow ASCII O(N) lookup loops
pub fn mtf_hash_name(name: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    
    for byte in name.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_hash_determinism() {
        let name = "blk.0.attn_q.weight";
        let h1 = mtf_hash_name(name);
        let h2 = mtf_hash_name(name);
        assert_eq!(h1, h2, "FNV-1a hash must be highly deterministic");
    }
}
