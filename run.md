
This guide explains how to download a model, compile it to the custom `.mtf` format, and run inference using the engine.

## Prerequisites

Before starting, ensure you have the required Python dependencies installed for downloading models, and that your Rust toolchain is up to date.

```bash
# Install Hugging Face Hub library for downloading models
pip install huggingface-hub
```

---

## Step 1: Download the Test Model

A Python helper script is provided in the `scripts` directory to automatically download the **Qwen2-0.5B** model (weights, config, and tokenizer) from the Hugging Face Hub.

Run the following command from the workspace root:

```bash
python scripts/download_test_model.py
```

The script will download the model files to `test_models/qwen2-0.5b/`.

---

## Step 2: Compile the Model to `.mtf` Format

The `mtf-compiler` converts standard Hugging Face Safetensors (which may contain `F16` or `BF16` weights) into the custom `.mtf` format with converted `F32` weights, sorting tensors by their name hashes for $O(\log N)$ binary search lookups [3.1.2, 5.1.2].

To compile the downloaded model, run:

```bash
cargo run --release --bin mtf-compiler -- \
  --input test_models/qwen2-0.5b \
  --output qwen2-0.5b.mtf \
  --verbose
```

### Options

* `--input` (or `-i`): The directory containing `model.safetensors`, `config.json`, and `tokenizer.json` [2.1.2].
* `--output` (or `-o`): The path where the compiled `.mtf` model will be saved (defaults to `model.mtf`) [2.1.2].
* `--verbose` (or `-v`): Increases verbosity. Use `-v` for standard logs, and `-vv` for detailed tensor layout information [2.1.2].

---

## Step 3: Run Inference with `mtf-engine`

Once compiled, you can load the `.mtf` file with the `mtf-engine`. The engine automatically extracts the tokenizer and configuration embedded in the file metadata and performs the autoregressive text generation [1.1.2, 4.1.2].

Run the following command to generate text:

```bash
cargo run --release --bin mtf-engine -- \
  --model qwen2-0.5b.mtf \
  --prompt "Once upon a time, in a land far away," \
  --max-tokens 50
```

### Options

* `--model` (or `-m`): Path to the compiled `.mtf` model [1.1.2].
* `--prompt` (or `-p`): The text prompt to feed into the model [1.1.2].
* `--max-tokens`: The maximum number of tokens to generate (defaults to `20`) [1.1.2].
* `--use-cuda`: Enable this flag if you have a CUDA-compatible GPU setup (`--use-cuda`) [1.1.2].

---

## Running Cargo Tests

To verify that the project components (hashing, math alignment, and conversions) are operating as expected, you can run the test suites:

```bash
cargo test --workspace
```
