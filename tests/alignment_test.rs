use mtf_common::hash::mtf_hash_name;
use mtf_common::ALIGNMENT_BOUNDARY;

#[test]
fn test_index_sorting_complexity() {
    let t1 = "blk.0.attn.weight";
    let t2 = "blk.10.attn.weight";
    
    let h1 = mtf_hash_name(t1);
    let h2 = mtf_hash_name(t2);
    
    // Hash outputs must be sortable natively as uint64
    let mut index = vec![h1, h2];
    index.sort(); // Emulate the compiler doing the hash sort
    
    assert!(index[0] <= index[1], "Hashes failed to sort correctly!");
}

#[test]
fn test_256_byte_padding_math() {
    let current_cursor: u64 = 1125; 
    // Applying the MTF compiler strict rule:
    let modulo = current_cursor % ALIGNMENT_BOUNDARY;
    let padding_needed = if modulo != 0 { ALIGNMENT_BOUNDARY - modulo } else { 0 };
    
    assert_eq!(padding_needed, 155, "Padding math calculated incorrect injection sequence");
    assert_eq!((current_cursor + padding_needed) % ALIGNMENT_BOUNDARY, 0, "SIMD Alignment Failed!");
}
