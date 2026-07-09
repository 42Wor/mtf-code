mod engine;
mod varbuilder;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use engine::MtfEngine;
use std::io::Write; // Needed for real-time stdout flushing
use std::path::PathBuf;
use std::time::{Duration, Instant}; // Needed for performance timing
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

    #[arg(short = 'n', long, default_value = "50")]
    max_tokens: usize,

    #[arg(long)]
    use_cuda: bool,

    /// Temperature parameter for generation (0.0 for greedy search)
    #[arg(short = 't', long, default_value = "0.0")]
    temperature: f64,

    /// Nucleus sampling top-p threshold
    #[arg(long, default_value = "1.0")]
    top_p: f64,

    /// Random seed for reproducible generation
    #[arg(long, default_value = "299792458")]
    seed: u64,

    #[arg(short = 'v', long, action = clap::ArgAction::Count)]
    verbose: u8,
}

/// Unified abstraction for supported model architectures
pub enum ModelArchitecture {
    Llama(Llama, LlamaCache),
    Qwen2(Qwen2),
}

impl ModelArchitecture {
    /// Dispatches the forward pass to the underlying architecture
    pub fn forward(&mut self, input: &Tensor, pos: usize) -> Result<Tensor> {
        match self {
            ModelArchitecture::Llama(model, cache) => model.forward(input, pos, cache).map_err(Into::into),
            ModelArchitecture::Qwen2(model) => model.forward(input, pos).map_err(Into::into),
        }
    }
}

pub struct Generator {
    model: ModelArchitecture,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
}

impl Generator {
    pub fn new(
        model: ModelArchitecture,
        tokenizer: Tokenizer,
        device: Device,
        temp: Option<f64>,
        top_p: Option<f64>,
        seed: u64,
    ) -> Result<Self> {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Ok(Self {
            model,
            device,
            tokenizer,
            logits_processor,
        })
    }

    pub fn generate(&mut self, prompt: &str, max_tokens: u64) -> Result<()> {
        let start_time = Instant::now();

        let mut tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec();

        let prompt_len = tokens.len();

        println!("\n=== Prompt Evaluation ===");
        println!("Prompt tokens: {}", prompt_len);
        println!("\n=== Generated Output ===");

        let mut generated_count = 0;
        let mut ttft = Duration::from_secs(0);

        for index in 0..max_tokens {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;

            // Pass execution to our unified dispatcher
            let logits = self.model.forward(&input, start_pos)?;

            // Extract the 1D logits slice [vocab_size] for the last token in the sequence
            let logits_seq_len = logits.dim(1)?;
            let logits = logits.i((0, logits_seq_len - 1))?.to_dtype(DType::F32)?;

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_count += 1;

            // Capture the time taken to produce the first token (TTFT)
            if index == 0 {
                ttft = start_time.elapsed();
            }

            let token_str = self
                .tokenizer
                .decode(&[next_token], true)
                .map_err(|e| anyhow::anyhow!(e))?;

            // Stream the generated token directly to stdout
            print!("{}", token_str);
            std::io::stdout().flush()?;
        }

        let total_duration = start_time.elapsed();
        let decode_duration = total_duration.saturating_sub(ttft);

        // Calculate decode speed (excluding the prefill step)
        let decode_tps = if decode_duration.as_secs_f64() > 0.0 {
            (generated_count - 1) as f64 / decode_duration.as_secs_f64()
        } else {
            0.0
        };

        // Calculate total throughput
        let overall_tps = generated_count as f64 / total_duration.as_secs_f64();

        println!("\n\n=== Performance Metrics ===");
        println!("Time to First Token (TTFT): {:.2?}", ttft);
        println!("Total Generation Time:     {:.2?}", total_duration);
        println!("Tokens Generated:          {}", generated_count);
        println!("Decode Speed:              {:.2} tokens/sec", decode_tps);
        println!("Overall Throughput:        {:.2} tokens/sec", overall_tps);

        Ok(())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let level = match args.verbose {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        _ => log::LevelFilter::Debug,
    };
    env_logger::Builder::new().filter_level(level).init();

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

    // 3. Polymorphically instantiate the model based on config type
    let model = match model_type {
        "qwen2" => {
            let config: Qwen2Config = serde_json::from_value(config_json)?;
            let qwen_model = Qwen2::new(&config, vb)?;
            ModelArchitecture::Qwen2(qwen_model)
        }
        "llama" => {
            // LlamaConfig is the deserializable configuration struct
            let config: LlamaConfig = serde_json::from_value(config_json)?;
            let resolved_config = config.into_config(false); // Disable flash attention for compatibility

            // Llama uses external Cache states (disable flash attention inside the cache state)
            let cache = LlamaCache::new(
                false,
                DType::F32,
                &resolved_config,
                &device,
            )?;
            let llama_model = Llama::load(vb, &resolved_config)?;
            ModelArchitecture::Llama(llama_model, cache)
        }
        _ => anyhow::bail!("Unsupported model type: {}", model_type),
    };

    let mut gen = Generator::new(
        model,
        tokenizer,
        device,
        Some(args.temperature),
        Some(args.top_p),
        args.seed,
    )?;

    gen.generate(&args.prompt, args.max_tokens as u64)?;

    Ok(())
}
