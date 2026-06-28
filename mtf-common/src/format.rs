#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
}
