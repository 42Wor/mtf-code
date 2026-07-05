use std::fs;
use std::path::Path;

pub fn build_metadata_json(input_dir: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let config_path = input_dir.join("config.json");
    let tokenizer_path = input_dir.join("tokenizer.json");

    let mut metadata = serde_json::json!({});

    if config_path.exists() {
        println!("[+] Embedding config.json...");
        let config_str = fs::read_to_string(&config_path)?;
        if let Ok(config_json) = serde_json::from_str::<serde_json::Value>(&config_str) {
            metadata["config"] = config_json;
        }
    } else {
        println!("[-] Warning: config.json not found in input directory!");
    }

    if tokenizer_path.exists() {
        println!("[+] Embedding tokenizer.json...");
        let tok_str = fs::read_to_string(&tokenizer_path)?;
        if let Ok(tok_json) = serde_json::from_str::<serde_json::Value>(&tok_str) {
            metadata["tokenizer"] = tok_json;
        }
    } else {
        println!("[-] Warning: tokenizer.json not found in input directory!");
    }

    Ok(metadata.to_string())
}
