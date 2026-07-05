use anyhow::{Context, Result};
use tokenizers::Tokenizer;

pub fn load_tokenizer_from_metadata(metadata: &serde_json::Value) -> Result<Tokenizer> {
    let tokenizer_json = metadata["tokenizer"]
        .as_object()
        .context("tokenizer not found in metadata")?;
    let tokenizer_str = serde_json::to_string(tokenizer_json)?;
    let tokenizer = Tokenizer::from_str(&tokenizer_str)?;
    Ok(tokenizer)
}
