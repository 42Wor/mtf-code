use crate::engine::MtfEngine;
use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::var_builder::SimpleBackend;
use candle_nn::VarBuilder;

pub struct MtfBackend<'a> {
    engine: &'a MtfEngine,
}

impl<'a> SimpleBackend for MtfBackend<'a> {
    fn get(
        &self,
        _s: Shape,
        name: &str,
        _h: candle_nn::Init,
        _dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.engine
            .get_tensor(name)
            .map_err(|e| candle_core::Error::wrap(e))?
            .to_device(dev)
    }

    fn get_unchecked(
        &self,
        name: &str,
        _dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.engine
            .get_tensor(name)
            .map_err(|e| candle_core::Error::wrap(e))?
            .to_device(dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.engine.contains_tensor(name)
    }
}

pub fn create_mtf_var_builder<'a>(engine: &'a MtfEngine, device: Device) -> VarBuilder<'a> {
    let backend = MtfBackend { engine };
    VarBuilder::from_backend(Box::new(backend), DType::F32, device)
}
