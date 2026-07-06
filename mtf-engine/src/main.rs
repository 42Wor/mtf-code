mod engine;
mod varbuilder;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use engine::MtfEngine;
use std::path::PathBuf;
use tokenizers::Tokenizer;
use varbuilder::create_mtf_var_builder;

// Import the supported model architectures
use candle_transformers::models::llama::{Cache as LlamaCache, Llama, LlamaConfig};
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM as Qwen2};

#[derive(Parser)]
struct Args {
    #[arg(short, long)]
    model: PathBuf,

    #[arg(short, long)]
    prompt: String,

    #[arg(short, long, default_value = "50")]
    max_tokens: usize,

    #[arg(long)]
    use_cuda: bool,
}

/// Unified abstraction for supported model architectures
pub enum ModelArchitecture {
    Llama(Llama),
    Qwen2(Qwen2),
}

impl ModelArchitecture {
    /// Dispatches the forward pass to the underlying architecture
    pub fn forward(
        &mut self,
        input: &Tensor,
        pos: usize,
        cache: &mut Option<LlamaCache>,
    ) -> Result<Tensor> {
        match self {
            ModelArchitecture::Qwen2(model) => {
                let logits = model.forward(input, pos)?;
                Ok(logits)
            }
            ModelArchitecture::Llama(model) => {
                let llama_cache = cache
                    .as_mut()
                    .context("Llama requires an active KV Cache state")?;
                let logits = model.forward(input, pos, llama_cache)?;
                Ok(logits)
            }
        }
    }
}

pub struct Generator {
    model: ModelArchitecture,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    llama_cache: Option<LlamaCache>,
}

impl Generator {
    pub fn new(
        model: ModelArchitecture,
        tokenizer: Tokenizer,
        device: Device,
        llama_cache: Option<LlamaCache>,
        temp: Option<f64>,
        top_p: Option<f64>,
    ) -> Result<Self> {
        let logits_processor = LogitsProcessor::new(299792458, temp, top_p);
        Ok(Self {
            model,
            device,
            tokenizer,
            logits_processor,
            llama_cache,
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

            // Pass execution to our unified dispatcher
            let logits = self
                .model
                .forward(&input, start_pos, &mut self.llama_cache)?;

            // Extract the 1D logits slice [vocab_size] for the last token using the actual logits sequence length
            let logits_seq_len = logits.dim(1)?;
            let logits = logits.i((0, logits_seq_len - 1))?.to_dtype(DType::F32)?;

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

    // 1. Load Tokenizer
    let tokenizer_json = engine.get_metadata()["tokenizer"]
        .as_object()
        .context("Tokenizer not found in metadata")?;
    let tokenizer_str = serde_json::to_string(tokenizer_json)?;
    let tokenizer = Tokenizer::from_bytes(tokenizer_str.as_bytes())
        .map_err(|e| anyhow::anyhow!(e))
        .context("Failed to parse tokenizer")?;

    // 2. Identify Model Type from Config Metadata
    let config_json = engine.get_config().clone();
    let model_type = config_json
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("llama"); // Fallback to Llama

    log::info!(
        "Dynamically configuring engine for architecture: {}",
        model_type
    );

    let vb = create_mtf_var_builder(&engine, device.clone());
    let mut llama_cache = None;

    // 3. Polymorphically instantiate the model based on config type
    let model = match model_type {
        "qwen2" => {
            let config: Qwen2Config = serde_json::from_value(config_json)?;
            let qwen_model = Qwen2::new(&config, vb)?;
            ModelArchitecture::Qwen2(qwen_model)
        }
        "llama" | _ => {
            // LlamaConfig is the deserializable configuration struct
            let config: LlamaConfig = serde_json::from_value(config_json)?;
            let resolved_config = config.into_config(false); // Disable flash attention for compatibility

            // Llama uses external Cache states (disable flash attention inside the cache state)
            llama_cache = Some(LlamaCache::new(
                false,
                DType::F32,
                &resolved_config,
                &device,
            )?);
            let llama_model = Llama::load(vb, &resolved_config)?;
            ModelArchitecture::Llama(llama_model)
        }
    };

    let mut gen = Generator::new(
        model,
        tokenizer,
        device,
        llama_cache,
        Some(0.0), // Greedy search
        Some(1.0),
    )?;

    let generated = gen.generate(&args.prompt, args.max_tokens as u64)?;
    println!("\nPrompt: {}\nGenerated:\n{}", args.prompt, generated);

    Ok(())
}
