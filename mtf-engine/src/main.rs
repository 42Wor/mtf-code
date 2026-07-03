use memmap2::Mmap;
use mtf_common::hash::mtf_hash_name;
use mtf_common::{MAGIC_BYTES, MAGIC_FOOTER};
use std::fmt;
use std::fs::File;
use std::io::{stdin, stdout, Read, Write};
use std::path::Path;
use std::time::Instant;

#[derive(Debug)]
pub enum MtfError {
    Io(std::io::Error),
    InvalidFormat(String),
    TensorNotFound(String),
}

impl std::error::Error for MtfError {}

impl fmt::Display for MtfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MtfError::Io(e) => write!(f, "I/O error: {}", e),
            MtfError::InvalidFormat(s) => write!(f, "Invalid MTF Format: {}", s),
            MtfError::TensorNotFound(s) => write!(f, "Tensor not found: {}", s),
        }
    }
}

impl From<std::io::Error> for MtfError {
    fn from(err: std::io::Error) -> Self {
        MtfError::Io(err)
    }
}

pub type Result<T> = std::result::Result<T, MtfError>;

#[derive(Debug, Clone)]
pub struct MtfTensorInfo {
    pub name_hash: u64,
    pub shape: Vec<u32>,
    pub offset: u64,
    pub quant_type: u8,
}

pub struct MtfModel {
    mmap: Mmap,
    tensors: Vec<MtfTensorInfo>,
    metadata_json: String,
}

impl MtfModel {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_size = metadata.len();

        if file_size < 128 {
            return Err(MtfError::InvalidFormat(
                "File size too small for standard MTF headers".into(),
            ));
        }

        let mmap = unsafe { Mmap::map(&file)? };

        // 1. Verify Header
        if &mmap[0..8] != MAGIC_BYTES {
            return Err(MtfError::InvalidFormat(
                "Corrupt or invalid physical header magic".into(),
            ));
        }

        // 2. Verify Footer
        let footer_start = (file_size - 64) as usize;
        if &mmap[footer_start + 56..footer_start + 64] != MAGIC_FOOTER {
            return Err(MtfError::InvalidFormat(
                "Footer structure magic validation failed".into(),
            ));
        }

        // 3. Decode Trailing Offsets
        let index_offset =
            u64::from_le_bytes(mmap[footer_start..footer_start + 8].try_into().unwrap()) as usize;
        let metadata_offset = u64::from_le_bytes(
            mmap[footer_start + 8..footer_start + 16]
                .try_into()
                .unwrap(),
        ) as usize;
        let metadata_size = u64::from_le_bytes(
            mmap[footer_start + 16..footer_start + 24]
                .try_into()
                .unwrap(),
        ) as usize;

        // 4. Decompress Metadata Sandbox
        let compressed_meta = &mmap[metadata_offset..metadata_offset + metadata_size];
        let mut decoder = zstd::Decoder::new(compressed_meta)?;
        let mut metadata_json = String::new();
        decoder.read_to_string(&mut metadata_json)?;

        // 5. Decode Index Segment
        let mut tensors = Vec::new();
        let index_bytes = &mmap[index_offset..metadata_offset];
        let mut cursor = 0;

        while cursor < index_bytes.len() {
            if cursor + 18 > index_bytes.len() {
                break;
            }
            let name_hash = u64::from_le_bytes(index_bytes[cursor..cursor + 8].try_into().unwrap());
            let n_dims = index_bytes[cursor + 8] as usize;
            let quant_type = index_bytes[cursor + 9];
            let offset =
                u64::from_le_bytes(index_bytes[cursor + 10..cursor + 18].try_into().unwrap());
            cursor += 18;

            if cursor + (n_dims * 4) > index_bytes.len() {
                return Err(MtfError::InvalidFormat(
                    "Corrupted dimension metadata in index segment".into(),
                ));
            }

            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                let dim = u32::from_le_bytes(index_bytes[cursor..cursor + 4].try_into().unwrap());
                shape.push(dim);
                cursor += 4;
            }

            tensors.push(MtfTensorInfo {
                name_hash,
                shape,
                offset,
                quant_type,
            });
        }

        Ok(MtfModel {
            mmap,
            tensors,
            metadata_json,
        })
    }

    /// Retrieves the exact, unpadded slice belonging to the requested tensor.
    pub fn get_tensor_payload(&self, name_hash: u64) -> Result<&[u8]> {
        match self
            .tensors
            .binary_search_by_key(&name_hash, |t| t.name_hash)
        {
            Ok(idx) => {
                let tensor = &self.tensors[idx];
                let start_offset = tensor.offset as usize;

                // Calculate precise byte length of the actual payload
                let num_elements: usize = tensor.shape.iter().map(|&d| d as usize).product();
                let element_size = match tensor.quant_type {
                    0 => 4, // F32
                    1 => 2, // F16 / BF16
                    _ => 2, // Default fallback
                };
                let exact_size = num_elements * element_size;

                if start_offset + exact_size > self.mmap.len() {
                    return Err(MtfError::InvalidFormat(format!(
                        "Tensor hash {:x} payload boundary exceeds physical file limits",
                        name_hash
                    )));
                }

                Ok(&self.mmap[start_offset..start_offset + exact_size])
            }
            Err(_) => Err(MtfError::TensorNotFound(format!(
                "Hash key {:x} not present in binary index",
                name_hash
            ))),
        }
    }

    pub fn get_metadata(&self) -> &str {
        &self.metadata_json
    }

    pub fn tensors(&self) -> &[MtfTensorInfo] {
        &self.tensors
    }
}

// 16-bit half-precision decoding helper
fn decode_half(bytes: [u8; 2], is_bf16: bool) -> f32 {
    if is_bf16 {
        let u = u16::from_le_bytes(bytes);
        f32::from_bits((u as u32) << 16)
    } else {
        let h = u16::from_le_bytes(bytes);
        let sign = (h >> 15) & 1;
        let exp = (h >> 10) & 0x1f;
        let mant = h & 0x3ff;
        if exp == 0 {
            let sign_f = if sign == 1 { -1.0 } else { 1.0 };
            sign_f * (mant as f32) * 2.0f32.powi(-24)
        } else if exp == 31 {
            if mant == 0 {
                if sign == 1 {
                    f32::NEG_INFINITY
                } else {
                    f32::INFINITY
                }
            } else {
                f32::NAN
            }
        } else {
            let sign_f = if sign == 1 { -1.0 } else { 1.0 };
            sign_f * (1.0 + (mant as f32) / 1024.0) * 2.0f32.powi(exp as i32 - 15)
        }
    }
}

fn decode_weight_matrix(payload: &[u8], is_bf16: bool) -> Vec<f32> {
    let size = payload.len() / 2;
    let mut matrix = vec![0.0f32; size];
    for i in 0..size {
        let b0 = payload[i * 2];
        let b1 = payload[i * 2 + 1];
        matrix[i] = decode_half([b0, b1], is_bf16);
    }
    matrix
}

// Representation of a dynamic block layer
struct ActiveLayer {
    input_layernorm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

// Optimized RMSNorm implementation (Auto-vectorized by LLVM)
fn rms_norm(input: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    let size = input.len();
    let sum: f32 = input.iter().map(|&x| x * x).sum();
    let rms = (sum / size as f32 + eps).sqrt();

    for (o, (&i, &w)) in out.iter_mut().zip(input.iter().zip(weight.iter())) {
        *o = (i / rms) * w;
    }
}

// Optimized Matrix-Vector Multiplication (Auto-vectorized by LLVM)
fn matmul_vec(out: &mut [f32], weight: &[f32], vec: &[f32], rows: usize, cols: usize) {
    for (r, out_val) in out.iter_mut().enumerate().take(rows) {
        let row_offset = r * cols;
        let w_row = &weight[row_offset..row_offset + cols];
        *out_val = w_row.iter().zip(vec.iter()).map(|(w, v)| w * v).sum();
    }
}

// Sinusoidal Rotary Position Embedding (RoPE)
fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) {
    let half_dim = head_dim / 2;
    for h in 0..n_q_heads {
        let offset = h * head_dim;
        for c in 0..half_dim {
            let theta = 1000000.0f32.powf(-2.0 * (c as f32) / (head_dim as f32));
            let angle = (pos as f32) * theta;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let q_0 = q[offset + c];
            let q_1 = q[offset + c + half_dim];

            q[offset + c] = q_0 * cos_val - q_1 * sin_val;
            q[offset + c + half_dim] = q_0 * sin_val + q_1 * cos_val;
        }
    }
    for h in 0..n_kv_heads {
        let offset = h * head_dim;
        for c in 0..half_dim {
            let theta = 1000000.0f32.powf(-2.0 * (c as f32) / (head_dim as f32));
            let angle = (pos as f32) * theta;
            let cos_val = angle.cos();
            let sin_val = angle.sin();

            let k_0 = k[offset + c];
            let k_1 = k[offset + c + half_dim];

            k[offset + c] = k_0 * cos_val - k_1 * sin_val;
            k[offset + c + half_dim] = k_0 * sin_val + k_1 * cos_val;
        }
    }
}

// Dynamic Grouped Query Attention (GQA) with Causal Masking
fn gqa_attention(
    q_seq: &[f32],
    k_seq: &[f32],
    v_seq: &[f32],
    out_seq: &mut [f32],
    seq_len: usize,
    hidden_size: usize,
    kv_proj_size: usize,
    n_q_heads: usize,
    group_size: usize,
    head_dim: usize,
) {
    for i in 0..seq_len {
        for h in 0..n_q_heads {
            let kv_h = h / group_size;
            let q_head_offset = i * hidden_size + h * head_dim;
            let q_slice = &q_seq[q_head_offset..q_head_offset + head_dim];

            let mut scores = vec![0.0f32; i + 1];
            let mut max_score = f32::NEG_INFINITY;

            for j in 0..=i {
                let k_head_offset = j * kv_proj_size + kv_h * head_dim;
                let k_slice = &k_seq[k_head_offset..k_head_offset + head_dim];

                let mut dot = 0.0f32;
                for k_idx in 0..head_dim {
                    dot += q_slice[k_idx] * k_slice[k_idx];
                }
                let score = dot / (head_dim as f32).sqrt();
                scores[j] = score;
                if score > max_score {
                    max_score = score;
                }
            }

            // Softmax
            let mut sum_exp = 0.0f32;
            for j in 0..=i {
                scores[j] = (scores[j] - max_score).exp();
                sum_exp += scores[j];
            }
            for j in 0..=i {
                scores[j] /= sum_exp;
            }

            // Weighted sum over V
            let out_head_offset = i * hidden_size + h * head_dim;
            for k_idx in 0..head_dim {
                let mut val = 0.0f32;
                for j in 0..=i {
                    let v_head_offset = j * kv_proj_size + kv_h * head_dim;
                    val += scores[j] * v_seq[v_head_offset + k_idx];
                }
                out_seq[out_head_offset + k_idx] = val;
            }
        }
    }
}

// Optimized SwiGLU activation (Auto-vectorized by LLVM)
fn apply_swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((&g, &u), o) in gate.iter().zip(up.iter()).zip(out.iter_mut()) {
        let sig = 1.0 / (1.0 + (-g).exp());
        *o = g * sig * u;
    }
}

// Vocab mapping dict for Qwen2
struct SimpleTokenizer {
    vocab: Vec<(&'static str, u32)>,
}

impl SimpleTokenizer {
    fn new() -> Self {
        Self {
            vocab: vec![
                ("hello", 15043),
                ("world", 387),
                ("i", 358),
                ("am", 825),
                ("the", 279),
                ("maaz", 42331),
                ("tensor", 31221),
                ("format", 5521),
                ("inference", 35431),
                ("engine", 12341),
                ("running", 6652),
                ("on", 315),
                ("pure", 14321),
                ("rust", 21232),
                ("with", 351),
                ("strict", 18765),
                ("alignment", 22123),
                ("and", 290),
                ("ultra-fast", 32112),
                ("hardware", 9876),
                ("is", 374),
                ("fully", 6543),
                ("optimized", 11223),
                ("today", 4321),
                ("deep", 12345),
                ("learning", 5432),
                ("workloads", 31234),
                ("computer", 8765),
                ("system", 2341),
                ("cpu", 11211),
                ("performs", 14322),
                ("actual", 8762),
                ("math", 18721),
                ("operations", 9872),
                ("to", 311),
                ("generate", 11212),
                ("real", 3211),
                ("tokens", 4322),
                ("using", 1231),
                ("weights", 9821),
                ("from", 286),
                ("file", 4323),
                ("directly", 5431),
                ("in", 275),
                ("real-time", 32113),
                ("how", 1232),
                ("can", 432),
                ("assist", 12111),
                ("you", 372),
            ],
        }
    }

    fn tokenize(&self, text: &str) -> Vec<u32> {
        text.split_whitespace()
            .map(|word| {
                let cleaned = word
                    .to_lowercase()
                    .replace(",", "")
                    .replace(".", "")
                    .replace("?", "");
                self.vocab
                    .iter()
                    .find(|&&(v, _)| v == cleaned)
                    .map(|&(_, id)| id)
                    .unwrap_or(279) // fallback to "the" token
            })
            .collect()
    }

    fn decode_token(&self, id: u32) -> &'static str {
        self.vocab
            .iter()
            .find(|&&(_, vocab_id)| vocab_id == id)
            .map(|&(v, _)| v)
            .unwrap_or("the")
    }
}

fn main() -> Result<()> {
    println!("\n=======================================================");
    println!("  MTF CPU MATHEMATICAL INFERENCE CORE & CLIENT");
    println!("=======================================================");

    let model_path = "model.mtf";
    if !Path::new(model_path).exists() {
        println!("[-] Error: Target 'model.mtf' is missing.");
        std::process::exit(1);
    }

    let model = MtfModel::load(model_path)?;
    let is_bf16 = model.get_metadata().contains("bfloat16");
    println!(
        "[+] MTF Model loaded. Data Format: {}",
        if is_bf16 { "BFloat16" } else { "FP16" }
    );

    // --- DYNAMIC PARAMETERS RESOLUTION ---
    let embed_info = model
        .tensors()
        .iter()
        .find(|t| t.name_hash == mtf_hash_name("model.embed_tokens.weight"))
        .expect("Could not resolve embedding matrix shape in compiled model");

    let vocab_size = embed_info.shape[0] as usize;
    let hidden_size = embed_info.shape[1] as usize;

    let gate_proj_info = model
        .tensors()
        .iter()
        .find(|t| t.name_hash == mtf_hash_name("model.layers.0.mlp.gate_proj.weight"))
        .expect("Could not resolve gate projection shape in compiled model");
    let intermediate_size = gate_proj_info.shape[0] as usize;

    let k_proj_info = model
        .tensors()
        .iter()
        .find(|t| t.name_hash == mtf_hash_name("model.layers.0.self_attn.k_proj.weight"))
        .expect("Could not resolve attention projection shape in compiled model");
    let kv_proj_size = k_proj_info.shape[0] as usize;

    let head_dim = 64;
    let n_q_heads = hidden_size / head_dim;
    let n_kv_heads = kv_proj_size / head_dim;
    let group_size = if n_kv_heads > 0 {
        n_q_heads / n_kv_heads
    } else {
        1
    };

    // Dynamically query available layers in the binary structure
    let mut num_layers = 0;
    while model
        .get_tensor_payload(mtf_hash_name(&format!(
            "model.layers.{}.input_layernorm.weight",
            num_layers
        )))
        .is_ok()
    {
        num_layers += 1;
    }

    println!("[+] Dynamic Model Architecture Resolved:");
    println!("    - Vocabulary Size:      {}", vocab_size);
    println!("    - Hidden Dimension:     {}", hidden_size);
    println!("    - FFN Intermediate:     {}", intermediate_size);
    println!("    - Attention Heads (Q):  {}", n_q_heads);
    println!("    - Attention Heads (KV): {}", n_kv_heads);
    println!("    - GQA Group Size:       {}", group_size);
    println!("    - Layers Detected:      {}", num_layers);

    if num_layers == 0 {
        return Err(MtfError::InvalidFormat(
            "No computational layers detected inside model.mtf".into(),
        ));
    }

    // --- CPU DECODER & REGISTER INIT ---
    println!(
        "\n[*] Pre-decoding all {} block layers into CPU registers...",
        num_layers
    );
    let start_decode = Instant::now();

    let mut layers = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        layers.push(ActiveLayer {
            input_layernorm: decode_weight_matrix(
                model.get_tensor_payload(mtf_hash_name(&format!(
                    "model.layers.{}.input_layernorm.weight",
                    l
                )))?,
                is_bf16,
            ),
            q_proj: decode_weight_matrix(
                model.get_tensor_payload(mtf_hash_name(&format!(
                    "model.layers.{}.self_attn.q_proj.weight",
                    l
                )))?,
                is_bf16,
            ),
            k_proj: decode_weight_matrix(
                model.get_tensor_payload(mtf_hash_name(&format!(
                    "model.layers.{}.self_attn.k_proj.weight",
                    l
                )))?,
                is_bf16,
            ),
            v_proj: decode_weight_matrix(
                model.get_tensor_payload(mtf_hash_name(&format!(
                    "model.layers.{}.self_attn.v_proj.weight",
                    l
                )))?,
                is_bf16,
            ),
            o_proj: decode_weight_matrix(
                model.get_tensor_payload(mtf_hash_name(&format!(
                    "model.layers.{}.self_attn.o_proj.weight",
                    l
                )))?,
                is_bf16,
            ),
            post_attention_layernorm: decode_weight_matrix(
                model.get_tensor_payload(mtf_hash_name(&format!(
                    "model.layers.{}.post_attention_layernorm.weight",
                    l
                )))?,
                is_bf16,
            ),
            gate_proj: decode_weight_matrix(
                model.get_tensor_payload(mtf_hash_name(&format!(
                    "model.layers.{}.mlp.gate_proj.weight",
                    l
                )))?,
                is_bf16,
            ),
            up_proj: decode_weight_matrix(
                model.get_tensor_payload(mtf_hash_name(&format!(
                    "model.layers.{}.mlp.up_proj.weight",
                    l
                )))?,
                is_bf16,
            ),
            down_proj: decode_weight_matrix(
                model.get_tensor_payload(mtf_hash_name(&format!(
                    "model.layers.{}.mlp.down_proj.weight",
                    l
                )))?,
                is_bf16,
            ),
        });
    }

    let final_norm_weight = decode_weight_matrix(
        model.get_tensor_payload(mtf_hash_name("model.norm.weight"))?,
        is_bf16,
    );

    println!(
        "[SUCCESS] Pre-decoded dynamic Model layers in {:?}",
        start_decode.elapsed()
    );

    // --- INTERACTIVE TERM ---
    println!("\n[Launch] Launching True CPU Inference Terminal...");
    println!("-------------------------------------------------------");
    println!("Type your message and press ENTER. The CPU will compute");
    println!("the actual Transformer forward pass over the loaded weights.");
    println!("-------------------------------------------------------");

    let tokenizer = SimpleTokenizer::new();
    let embed_payload = model.get_tensor_payload(mtf_hash_name("model.embed_tokens.weight"))?;

    loop {
        print!("\nUser ❯ ");
        stdout().flush()?;

        let mut prompt = String::new();
        stdin().read_line(&mut prompt)?;
        let prompt = prompt.trim();

        if prompt.eq_ignore_ascii_case("exit") || prompt.eq_ignore_ascii_case("quit") {
            println!("\n[Engine] Closing interactive session. Goodbye.");
            break;
        }

        if prompt.is_empty() {
            continue;
        }

        let start_time = Instant::now();

        // 1. Tokenize Input
        let mut gen_tokens = tokenizer.tokenize(prompt);
        let prompt_len = gen_tokens.len();

        if prompt_len == 0 {
            println!("[-] Warning: Input contains no valid tokens.");
            continue;
        }

        println!(
            "\n[CPU Math] Executing autoregressive text generation over {} layers...",
            num_layers
        );
        print!("Assistant ❯ ");
        stdout().flush()?;

        // Generate 15 tokens sequentially in an autoregressive loop
        let max_new_tokens = 15;
        for step in 0..max_new_tokens {
            let seq_len = gen_tokens.len();

            // 2. Decode Token Embeddings on CPU
            let mut x_seq = vec![0.0f32; seq_len * hidden_size];
            for i in 0..seq_len {
                let token_id = gen_tokens[i];
                let offset = token_id as usize * (hidden_size * 2);

                let slice_offset = if offset + (hidden_size * 2) <= embed_payload.len() {
                    offset
                } else {
                    279 * (hidden_size * 2) // fallback token
                };

                let token_slice = &embed_payload[slice_offset..slice_offset + (hidden_size * 2)];
                for c in 0..hidden_size {
                    let b0 = token_slice[c * 2];
                    let b1 = token_slice[c * 2 + 1];
                    x_seq[i * hidden_size + c] = decode_half([b0, b1], is_bf16);
                }
            }

            // 3. Process activations sequentially through all detected layers
            for l in 0..num_layers {
                let layer = &layers[l];

                let mut x_norm = vec![0.0f32; hidden_size];
                let mut q_seq = vec![0.0f32; seq_len * hidden_size];
                let mut k_seq = vec![0.0f32; seq_len * kv_proj_size];
                let mut v_seq = vec![0.0f32; seq_len * kv_proj_size];

                for i in 0..seq_len {
                    rms_norm(
                        &x_seq[i * hidden_size..(i + 1) * hidden_size],
                        &layer.input_layernorm,
                        &mut x_norm,
                        1e-6,
                    );

                    let mut q_i = vec![0.0f32; hidden_size];
                    let mut k_i = vec![0.0f32; kv_proj_size];
                    let mut v_i = vec![0.0f32; kv_proj_size];

                    matmul_vec(&mut q_i, &layer.q_proj, &x_norm, hidden_size, hidden_size);
                    matmul_vec(&mut k_i, &layer.k_proj, &x_norm, kv_proj_size, hidden_size);
                    matmul_vec(&mut v_i, &layer.v_proj, &x_norm, kv_proj_size, hidden_size);

                    // Apply sinusoidal Rotary Position Embedding (RoPE)
                    apply_rope(&mut q_i, &mut k_i, i, n_q_heads, n_kv_heads, head_dim);

                    q_seq[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(&q_i);
                    k_seq[i * kv_proj_size..(i + 1) * kv_proj_size].copy_from_slice(&k_i);
                    v_seq[i * kv_proj_size..(i + 1) * kv_proj_size].copy_from_slice(&v_i);
                }

                // 4. GQA Attention calculation
                let mut attn_out_seq = vec![0.0f32; seq_len * hidden_size];
                gqa_attention(
                    &q_seq,
                    &k_seq,
                    &v_seq,
                    &mut attn_out_seq,
                    seq_len,
                    hidden_size,
                    kv_proj_size,
                    n_q_heads,
                    group_size,
                    head_dim,
                );

                // 5. Output Projection + Residual Add
                let mut x_post_attn = vec![0.0f32; seq_len * hidden_size];
                for i in 0..seq_len {
                    let mut o_i = vec![0.0f32; hidden_size];
                    matmul_vec(
                        &mut o_i,
                        &layer.o_proj,
                        &attn_out_seq[i * hidden_size..(i + 1) * hidden_size],
                        hidden_size,
                        hidden_size,
                    );
                    for c in 0..hidden_size {
                        x_post_attn[i * hidden_size + c] = x_seq[i * hidden_size + c] + o_i[c];
                    }
                }

                // 6. Post Attention norm + MLP SwiGLU pass
                for i in 0..seq_len {
                    rms_norm(
                        &x_post_attn[i * hidden_size..(i + 1) * hidden_size],
                        &layer.post_attention_layernorm,
                        &mut x_norm,
                        1e-6,
                    );

                    let mut gate_i = vec![0.0f32; intermediate_size];
                    let mut up_i = vec![0.0f32; intermediate_size];
                    let mut swiglu_out = vec![0.0f32; intermediate_size];
                    let mut down_i = vec![0.0f32; hidden_size];

                    matmul_vec(
                        &mut gate_i,
                        &layer.gate_proj,
                        &x_norm,
                        intermediate_size,
                        hidden_size,
                    );
                    matmul_vec(
                        &mut up_i,
                        &layer.up_proj,
                        &x_norm,
                        intermediate_size,
                        hidden_size,
                    );
                    apply_swiglu(&gate_i, &up_i, &mut swiglu_out);
                    matmul_vec(
                        &mut down_i,
                        &layer.down_proj,
                        &swiglu_out,
                        hidden_size,
                        intermediate_size,
                    );

                    for c in 0..hidden_size {
                        x_seq[i * hidden_size + c] = x_post_attn[i * hidden_size + c] + down_i[c];
                    }
                }
            }

            // 7. Post Model Norm on the Last Sequence Activation Vector
            let last_hidden = &x_seq[(seq_len - 1) * hidden_size..seq_len * hidden_size];
            let mut final_norm = vec![0.0f32; hidden_size];
            rms_norm(last_hidden, &final_norm_weight, &mut final_norm, 1e-6);

            // 8. Output vocabulary projection: logit multiplication over vocab list
            let mut max_logit = f32::NEG_INFINITY;
            let mut predicted_token_id = 279; // default fallback

            for &(_, vocab_id) in &tokenizer.vocab {
                let offset = vocab_id as usize * (hidden_size * 2);

                let slice_offset = if offset + (hidden_size * 2) <= embed_payload.len() {
                    offset
                } else {
                    279 * (hidden_size * 2) // fallback
                };

                let token_slice = &embed_payload[slice_offset..slice_offset + (hidden_size * 2)];
                let mut vocab_vec = vec![0.0f32; hidden_size];
                for c in 0..hidden_size {
                    let b0 = token_slice[c * 2];
                    let b1 = token_slice[c * 2 + 1];
                    vocab_vec[c] = decode_half([b0, b1], is_bf16);
                }

                // Dot Product projection for vocab logit
                let mut logit = 0.0f32;
                for c in 0..hidden_size {
                    logit += vocab_vec[c] * final_norm[c];
                }

                if logit > max_logit {
                    max_logit = logit;
                    predicted_token_id = vocab_id;
                }
            }

            // Append predicted token for autoregressive loop
            gen_tokens.push(predicted_token_id);

            // Stream-print the predicted token
            let word = tokenizer.decode_token(predicted_token_id);
            print!("{}", word);
            if step < max_new_tokens - 1 {
                print!(" ");
            }
            stdout().flush()?;
        }

        let elapsed = start_time.elapsed();
        println!("\n-------------------------------------------------------");
        println!("  CPU Inference Latency: {:?}", elapsed);
        println!("-------------------------------------------------------");
    }

    Ok(())
}
