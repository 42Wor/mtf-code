mod compiler;
mod metadata;
mod tensor;
mod utils;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "MTF Compiler: Bundles weights, config, and tokenizer into a single .mtf file", long_about = None)]
struct Args {
    /// Input directory containing model.safetensors, config.json, and tokenizer.json
    #[arg(short, long)]
    input: PathBuf,

    /// Output MTF file path
    #[arg(short, long, default_value = "model.mtf")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    println!("\n[MZTensor Labs] Starting MTF Compiler Pipeline...");
    
    if !args.input.is_dir() {
        eprintln!("[-] Error: Input must be a directory containing the model files.");
        std::process::exit(1);
    }

    println!("[+] Input Directory: {:?}", args.input);
    println!("[+] Output File: {:?}", args.output);

    compiler::run_compile(&args.input, &args.output)?;
    
    Ok(())
}
