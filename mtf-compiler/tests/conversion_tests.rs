use mtf_compiler::utils::decode_half;

#[test]
fn test_f16_to_f32() {
    let bytes = [0x00, 0x3C];
    let val = decode_half(bytes, false);
    assert!((val - 1.0).abs() < 1e-6);
}

#[test]
fn test_bf16_to_f32() {
    let bytes = [0x80, 0x3F];
    let val = decode_half(bytes, true);
    assert!((val - 1.0).abs() < 1e-6);
}
