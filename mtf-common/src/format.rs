#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 4,
}
