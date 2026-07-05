use anyhow::Result;
use serde_json::json;
use std::fs;
use std::path::Path;

pub fn build_metadata_json(input_dir: &Path) -> Result<String> {
    let config_path = input_dir.join("config.json");
    let tokenizer_path = input_dir.join("tokenizer.json");

    let mut metadata = json!({});

    if config_path.exists() {
        log::info!("Embedding config.json");
        let config_str = fs::read_to_string(&config_path)?;
        if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_str) {
            metadata["config"] = config_json;
        } else {
            log::warn!("config.json exists but is not valid JSON – skipping");
        }
    } else {
        log::warn!("config.json not found in input directory!");
    }

    if tokenizer_path.exists() {
        log::info!("Embedding tokenizer.json");
        let tok_str = fs::read_to_string(&tokenizer_path)?;
        if let Ok(tok_json) = serde_json::from_str::<serde_json::Value>(&tok_str) {
            metadata["tokenizer"] = tok_json;
        } else {
            log::warn!("tokenizer.json exists but is not valid JSON – skipping");
        }
    } else {
        log::warn!("tokenizer.json not found in input directory!");
    }

    Ok(metadata.to_string())
}
