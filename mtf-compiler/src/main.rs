mod compiler;
mod metadata;
mod tensor;
mod utils;
mod validator;

use clap::Parser;
use log::LevelFilter;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "MTF Compiler", long_about = None)]
struct Args {
    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long, default_value = "model.mtf")]
    output: PathBuf,

    /// Increase verbosity (-v for info, -vv for debug)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    verbose: u8,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let level = match args.verbose {
        0 => LevelFilter::Warn,
        1 => LevelFilter::Info,
        _ => LevelFilter::Debug,
    };
    env_logger::Builder::new()
        .filter_level(level)
        .format_timestamp_millis()
        .init();

    log::info!("MZTensor Labs – MTF Compiler Pipeline");
    log::debug!("Input directory: {:?}", args.input);
    log::debug!("Output file: {:?}", args.output);

    if !args.input.is_dir() {
        anyhow::bail!("Input must be a directory containing the model files.");
    }

    compiler::run_compile(&args.input, &args.output)?;

    log::info!("Compilation completed successfully.");
    Ok(())
}
