use crate::utils::decode_half;
use rayon::prelude::*;

pub enum TensorData<'a> {
    Borrowed(&'a [u8]),
    Owned(Vec<u8>),
}

impl<'a> TensorData<'a> {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            TensorData::Borrowed(s) => s,
            TensorData::Owned(v) => v,
        }
    }
}

pub struct CompileTensor<'a> {
    pub hash: u64,
    pub name: String,
    pub shape: Vec<usize>,
    pub data_type: u8,
    pub raw_data: TensorData<'a>,
    pub absolute_offset: u64,
}

pub fn process_tensor_data<'a>(
    raw_data: &'a [u8],
    dtype: safetensors::Dtype,
) -> TensorData<'a> {
    match dtype {
        safetensors::Dtype::F32 => TensorData::Borrowed(raw_data),
        safetensors::Dtype::F16 | safetensors::Dtype::BF16 => {
            let is_bf16 = dtype == safetensors::Dtype::BF16;
            
            // PARALLEL CONVERSION: Split into 2-byte chunks and process across all CPU cores
            let converted: Vec<u8> = raw_data
                .par_chunks_exact(2)
                .flat_map_iter(|chunk| {
                    let val = decode_half([chunk[0], chunk[1]], is_bf16);
                    val.to_le_bytes()
                })
                .collect();
                
            TensorData::Owned(converted)
        }
        _ => TensorData::Borrowed(raw_data),
    }
}
