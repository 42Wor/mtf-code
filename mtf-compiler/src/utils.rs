pub fn decode_half(bytes: [u8; 2], is_bf16: bool) -> f32 {
    if is_bf16 {
        let u = u16::from_le_bytes(bytes);
        f32::from_bits((u as u32) << 16)
    } else {
        let h = u16::from_le_bytes(bytes);
        let sign = (h >> 15) & 1;
        let exp = (h >> 10) & 0x1f;
        let mant = h & 0x3ff;
        if exp == 0 {
            let sign_f = if sign == 1 { -1.0 } else { 1.0 };
            sign_f * (mant as f32) * 2.0f32.powi(-24)
        } else if exp == 31 {
            if mant == 0 {
                if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
            } else {
                f32::NAN
            }
        } else {
            let sign_f = if sign == 1 { -1.0 } else { 1.0 };
            sign_f * (1.0 + (mant as f32) / 1024.0) * 2.0f32.powi(exp as i32 - 15)
        }
    }
}
