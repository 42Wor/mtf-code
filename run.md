
# `run.md`

This guide explains how to download a model, compile it to the custom `.mtf` format, and run inference using the engine with performance metrics and custom sampling configurations.

## Prerequisites

Before starting, ensure you have the required Python dependencies installed to download Hugging Face models.

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

The `mtf-compiler` converts standard Hugging Face Safetensors (which may contain `F16` or `BF16` weights) into the custom `.mtf` format with converted `F32` weights, sorting tensors by their name hashes for O(1) hash map lookups [3.1.2, 5.1.2].

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

Once compiled, you can load the `.mtf` file with the `mtf-engine`. The engine automatically extracts the tokenizer and configuration embedded in the file metadata, streams token generation in real time, and measures hardware execution speeds.

### Running with Default Options (Greedy Search, 50 Tokens)

```bash
cargo run --release --bin mtf-engine -- \
  --model qwen2-0.5b.mtf \
  --prompt "Once upon a time, in a land far away,"
```

### Running with Custom Sampling Options (Creative Search, 100 Tokens)

```bash
cargo run --release --bin mtf-engine -- \
  --model qwen2-0.5b.mtf \
  --prompt "Explain quantum computing in one sentence:" \
  --max-tokens 100 \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 42
```

### All CLI Options for `mtf-engine`

* `--model` (or `-m`): Path to the compiled `.mtf` model file.
* `--prompt` (or `-p`): The text prompt to feed into the model.
* `--max-tokens` (or `-n`): The maximum number of tokens to generate (defaults to `50`).
* `--temperature` (or `-t`): Controls the randomness of generation. Set to `0.0` for greedy search (the default), and higher values (e.g., `0.7`) for more creative outputs.
* `--top-p`: Active during non-greedy sampling (`temp > 0`). Keeps only the top tokens whose cumulative probability exceeds the threshold (defaults to `1.0`).
* `--seed`: An unsigned 64-bit integer (`u64`) seed used to make non-greedy sampling reproducible.
* `--use-cuda`: Enable this flag if you have a CUDA-compatible GPU setup and wish to run matrix multiplication on the GPU.

---

## Expected Output Format

The engine now streams the generated text token-by-token and outputs a performance analysis at the end of execution:

```text
=== Prompt Evaluation ===
Prompt tokens: 11

=== Generated Output ===
 there was a man named John. He was a very good man, but he was also very poor...

=== Performance Metrics ===
Time to First Token (TTFT): 1.84s
Total Generation Time:     3.24s
Tokens Generated:          50
Decode Speed:              35.12 tokens/sec
Overall Throughput:        15.43 tokens/sec
```

---

## Running Cargo Tests

To verify that the project components (hashing, math alignment, and conversions) are operating as expected, you can run the test suites:

```bash
cargo test --workspace
```
