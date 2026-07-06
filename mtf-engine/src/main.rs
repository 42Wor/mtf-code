mod engine;
mod varbuilder;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, Llama, LlamaConfig};
use clap::Parser;
use engine::MtfEngine;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use varbuilder::create_mtf_var_builder;

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    model: PathBuf,

    #[arg(short, long)]
    prompt: String,

    #[arg(short, long, default_value = "20")]
    max_tokens: usize,

    #[arg(long)]
    use_cuda: bool,
}

pub struct Generator {
    model: Llama,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    cache: Cache,
}

impl Generator {
    pub fn new(
        model: Llama,
        config: &Config,
        tokenizer: Tokenizer,
        device: Device,
        temp: Option<f64>,
        top_p: Option<f64>,
    ) -> Result<Self> {
        let logits_processor = LogitsProcessor::new(299792458, temp, top_p);
        let cache = Cache::new(true, DType::F32, config, &device)?;
        Ok(Self {
            model,
            device,
            tokenizer,
            logits_processor,
            cache,
        })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: u64) -> Result<String> {
        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec();

        let mut generated = String::new();

        for index in 0..max_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward(&input, start_pos, &mut self.cache)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);

            let token_str = self
                .tokenizer
                .decode(&[next_token], true)
                .map_err(|e| anyhow::anyhow!(e))?;
            generated.push_str(&token_str);
        }

        Ok(generated)
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();

    let device = if args.use_cuda {
        Device::cuda_if_available(0)?
    } else {
        Device::Cpu
    };

    let engine = MtfEngine::load(&args.model)?;

    // Load tokenizer from metadata
    let tokenizer_json = engine.get_metadata()["tokenizer"]
        .as_object()
        .context("Tokenizer not found in metadata")?;
    let tokenizer_str = serde_json::to_string(tokenizer_json)?;

    // Convert tokenizer loading error type before appending anyhow context
    let tokenizer = Tokenizer::from_bytes(tokenizer_str.as_bytes())
        .map_err(|e| anyhow::anyhow!(e))
        .context("Failed to parse tokenizer from metadata")?;

    // Deserialize into LlamaConfig and safely convert into Llama::Config
    let config_json = engine.get_config().clone();
    let llama_config: LlamaConfig = serde_json::from_value(config_json)?;
    let config = llama_config.into_config(false); // Disables flash-attention to prevent potential device crashes

    let vb = create_mtf_var_builder(&engine, device.clone());
    let model = Llama::load(vb, &config)?;

    let mut gen = Generator::new(
        model,
        &config,
        tokenizer,
        device,
        Some(0.0), // Temperature (0.0 = greedy)
        Some(1.0), // Top‑p
    )?;
    let generated = gen.generate(&args.prompt, args.max_tokens as u64)?;

    println!("{}", generated);

    Ok(())
}
