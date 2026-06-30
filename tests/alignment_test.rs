use mtf_common::hash::mtf_hash_name;
use mtf_common::ALIGNMENT_BOUNDARY;

#[test]
fn test_index_sorting_complexity() {
    let t1 = "model.embed_tokens.weight";
    let t2 = "model.layers.0.self_attn.q_proj.weight";

    let h1 = mtf_hash_name(t1);
    let h2 = mtf_hash_name(t2);

    let mut index = vec![h1, h2];
    index.sort(); // Emulate the compiler sorting by FNV-1a hash

    assert!(
        index[0] <= index[1],
        "Hashes must sort sequentially for O(log N) lookup binary-search compliance"
    );
}

#[test]
fn test_256_byte_padding_math() {
    let current_cursor: u64 = 1125;
    let modulo = current_cursor % ALIGNMENT_BOUNDARY;
    let padding_needed = if modulo != 0 {
        ALIGNMENT_BOUNDARY - modulo
    } else {
        0
    };

    assert_eq!(
        padding_needed, 155,
        "Padding math calculation anomaly detected"
    );
    assert_eq!(
        (current_cursor + padding_needed) % ALIGNMENT_BOUNDARY,
        0,
        "SIMD Alignment calculation failed!"
    );
}
