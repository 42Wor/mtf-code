use memmap2::Mmap;
use mtf_common::hash::mtf_hash_name;
use mtf_common::{MAGIC_BYTES, MAGIC_FOOTER};
use rand::Rng;
use serde::Deserialize;
use std::fmt;
use std::fs::File;
use std::io::{stdin, stdout, Read, Write};
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

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
            return Err(MtfError::InvalidFormat("File size too small".into()));
        }

        let mmap = unsafe { Mmap::map(&file)? };

        if &mmap[0..8] != MAGIC_BYTES {
            return Err(MtfError::InvalidFormat("Corrupt header magic".into()));
        }

        let footer_start = (file_size - 64) as usize;
        if &mmap[footer_start + 56..footer_start + 64] != MAGIC_FOOTER {
            return Err(MtfError::InvalidFormat("Corrupt footer magic".into()));
        }

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

        let compressed_meta = &mmap[metadata_offset..metadata_offset + metadata_size];
        let mut decoder = zstd::Decoder::new(compressed_meta)?;
        let mut metadata_json = String::new();
        decoder.read_to_string(&mut metadata_json)?;

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
                return Err(MtfError::InvalidFormat("Corrupted dimensions".into()));
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

    pub fn get_tensor(&self, name_hash: u64) -> Result<(&[u8], u8)> {
        match self
            .tensors
            .binary_search_by_key(&name_hash, |t| t.name_hash)
        {
            Ok(idx) => {
                let tensor = &self.tensors[idx];
                let start_offset = tensor.offset as usize;
                let num_elements: usize = tensor.shape.iter().map(|&d| d as usize).product();
                let element_size = match tensor.quant_type {
                    0 => 4,
                    1 => 2,
                    _ => 2,
                };
                let exact_size = num_elements * element_size;

                if start_offset + exact_size > self.mmap.len() {
                    return Err(MtfError::InvalidFormat(
                        "Payload exceeds file limits".into(),
                    ));
                }
                Ok((
                    &self.mmap[start_offset..start_offset + exact_size],
                    tensor.quant_type,
                ))
            }
            Err(_) => Err(MtfError::TensorNotFound(format!(
                "Hash {:x} not found",
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

// Standard IEEE-754 F16 Decoder
fn decode_f16(bytes: [u8; 2]) -> f32 {
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

fn decode_weight_matrix(payload: &[u8], quant_type: u8) -> Vec<f32> {
    if quant_type == 0 {
        let size = payload.len() / 4;
        let mut matrix = vec![0.0f32; size];
        for i in 0..size {
            let chunk = [
                payload[i * 4],
                payload[i * 4 + 1],
                payload[i * 4 + 2],
                payload[i * 4 + 3],
            ];
            matrix[i] = f32::from_le_bytes(chunk);
        }
        matrix
    } else {
        let size = payload.len() / 2;
        let mut matrix = vec![0.0f32; size];
        for i in 0..size {
            let b0 = payload[i * 2];
            let b1 = payload[i * 2 + 1];
            matrix[i] = decode_f16([b0, b1]);
        }
        matrix
    }
}

/// Helper to safely fetch optional tensors (like biases)
fn get_optional_tensor(model: &MtfModel, name: &str) -> Result<Option<Vec<f32>>> {
    match model.get_tensor(mtf_hash_name(name)) {
        Ok((payload, qtype)) => Ok(Some(decode_weight_matrix(payload, qtype))),
        Err(MtfError::TensorNotFound(_)) => Ok(None),
        Err(e) => Err(e),
    }
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub model_type: Option<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub rms_norm_eps: Option<f32>,
    pub rope_theta: Option<f32>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl ModelConfig {
    pub fn get_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
    pub fn get_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

struct ActiveLayer {
    input_layernorm: Vec<f32>,
    q_proj: Vec<f32>,
    q_bias: Option<Vec<f32>>,
    k_proj: Vec<f32>,
    k_bias: Option<Vec<f32>>,
    v_proj: Vec<f32>,
    v_bias: Option<Vec<f32>>,
    o_proj: Vec<f32>,
    post_attention_layernorm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

fn rms_norm(input: &[f32], weight: &[f32], out: &mut [f32], eps: f32) {
    let size = input.len();
    let sum: f32 = input.iter().map(|&x| x * x).sum();
    let rms = (sum / size as f32 + eps).sqrt();
    for (o, (&i, &w)) in out.iter_mut().zip(input.iter().zip(weight.iter())) {
        *o = (i / rms) * w;
    }
}

fn matmul_vec(out: &mut [f32], weight: &[f32], vec: &[f32], rows: usize, cols: usize) {
    for (r, out_val) in out.iter_mut().enumerate().take(rows) {
        let row_offset = r * cols;
        let w_row = &weight[row_offset..row_offset + cols];
        *out_val = w_row.iter().zip(vec.iter()).map(|(w, v)| w * v).sum();
    }
}

fn matmul_vec_bias(
    out: &mut [f32],
    weight: &[f32],
    vec: &[f32],
    bias: &[f32],
    rows: usize,
    cols: usize,
) {
    for (r, out_val) in out.iter_mut().enumerate().take(rows) {
        let row_offset = r * cols;
        let w_row = &weight[row_offset..row_offset + cols];
        *out_val = w_row
            .iter()
            .zip(vec.iter())
            .map(|(w, v)| w * v)
            .sum::<f32>()
            + bias[r];
    }
}

fn apply_rope(
    q: &mut [f32],
    k: &mut [f32],
    pos: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    theta_base: f32,
) {
    let half_dim = head_dim / 2;
    for h in 0..n_q_heads {
        let offset = h * head_dim;
        for c in 0..half_dim {
            let theta = theta_base.powf(-2.0 * (c as f32) / (head_dim as f32));
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
            let theta = theta_base.powf(-2.0 * (c as f32) / (head_dim as f32));
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

            let mut sum_exp = 0.0f32;
            for j in 0..=i {
                scores[j] = (scores[j] - max_score).exp();
                sum_exp += scores[j];
            }
            for j in 0..=i {
                scores[j] /= sum_exp;
            }

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

fn apply_swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((&g, &u), o) in gate.iter().zip(up.iter()).zip(out.iter_mut()) {
        let sig = 1.0 / (1.0 + (-g).exp());
        *o = g * sig * u;
    }
}

// Real BPE tokenizer wrapper
struct DynamicTokenizer {
    tokenizer: Tokenizer,
}

impl DynamicTokenizer {
    fn new(metadata_json: &str) -> Self {
        let meta: serde_json::Value =
            serde_json::from_str(metadata_json).expect("Failed to parse metadata JSON");
        let tokenizer_json = meta["tokenizer"].to_string();
        let tokenizer = Tokenizer::from_bytes(tokenizer_json.as_bytes())
            .expect("Failed to load tokenizer from metadata");
        Self { tokenizer }
    }

    fn tokenize(&self, text: &str) -> Vec<u32> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .expect("Tokenization failed");
        encoding.get_ids().to_vec()
    }

    fn decode_token(&self, id: u32) -> String {
        self.tokenizer
            .id_to_token(id)
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("<unk:{}>", id))
    }
}

fn main() -> Result<()> {
    println!("\n=======================================================");
    println!("  MTF CPU MATHEMATICAL INFERENCE CORE & CLIENT (F32)");
    println!("=======================================================");

    let model_path = "model.mtf";
    if !Path::new(model_path).exists() {
        println!("[-] Error: Target 'model.mtf' is missing.");
        std::process::exit(1);
    }

    let model = MtfModel::load(model_path)?;
    println!("[+] MTF Model loaded successfully.");

    // --- Phase 1: Dynamic Configuration Parsing ---
    let meta: serde_json::Value =
        serde_json::from_str(model.get_metadata()).expect("Failed to parse metadata JSON");

    let config: ModelConfig = serde_json::from_value(meta["config"].clone())
        .expect("Failed to parse ModelConfig from metadata");

    let vocab_size = config.vocab_size;
    let hidden_size = config.hidden_size;
    let intermediate_size = config.intermediate_size;
    let num_layers = config.num_hidden_layers;
    let n_q_heads = config.num_attention_heads;
    let n_kv_heads = config.get_kv_heads();
    let head_dim = config.get_head_dim();
    let group_size = if n_kv_heads > 0 {
        n_q_heads / n_kv_heads
    } else {
        1
    };

    let rms_norm_eps = config.rms_norm_eps.unwrap_or(1e-6);
    let rope_theta = config.rope_theta.unwrap_or(10000.0);
    let kv_proj_size = n_kv_heads * head_dim;

    println!("[+] Dynamic Model Architecture Resolved:");
    println!(
        "    - Model Type:           {}",
        config.model_type.as_deref().unwrap_or("Unknown")
    );
    println!("    - Vocabulary Size:      {}", vocab_size);
    println!("    - Hidden Dimension:     {}", hidden_size);
    println!("    - FFN Intermediate:     {}", intermediate_size);
    println!("    - Attention Heads (Q):  {}", n_q_heads);
    println!("    - Attention Heads (KV): {}", n_kv_heads);
    println!("    - Head Dimension:       {}", head_dim);
    println!("    - GQA Group Size:       {}", group_size);
    println!("    - RMS Norm Epsilon:     {}", rms_norm_eps);
    println!("    - RoPE Theta:           {}", rope_theta);
    println!("    - Layers Detected:      {}", num_layers);

    println!(
        "\n[*] Pre-decoding all {} block layers into CPU registers...",
        num_layers
    );
    let start_decode = Instant::now();

    let mut layers = Vec::with_capacity(num_layers);
    for l in 0..num_layers {
        let (in_norm_p, in_norm_q) = model.get_tensor(mtf_hash_name(&format!(
            "model.layers.{}.input_layernorm.weight",
            l
        )))?;
        let (q_proj_p, q_proj_q) = model.get_tensor(mtf_hash_name(&format!(
            "model.layers.{}.self_attn.q_proj.weight",
            l
        )))?;
        let q_bias =
            get_optional_tensor(&model, &format!("model.layers.{}.self_attn.q_proj.bias", l))?;

        let (k_proj_p, k_proj_q) = model.get_tensor(mtf_hash_name(&format!(
            "model.layers.{}.self_attn.k_proj.weight",
            l
        )))?;
        let k_bias =
            get_optional_tensor(&model, &format!("model.layers.{}.self_attn.k_proj.bias", l))?;

        let (v_proj_p, v_proj_q) = model.get_tensor(mtf_hash_name(&format!(
            "model.layers.{}.self_attn.v_proj.weight",
            l
        )))?;
        let v_bias =
            get_optional_tensor(&model, &format!("model.layers.{}.self_attn.v_proj.bias", l))?;

        let (o_proj_p, o_proj_q) = model.get_tensor(mtf_hash_name(&format!(
            "model.layers.{}.self_attn.o_proj.weight",
            l
        )))?;
        let (post_norm_p, post_norm_q) = model.get_tensor(mtf_hash_name(&format!(
            "model.layers.{}.post_attention_layernorm.weight",
            l
        )))?;
        let (gate_proj_p, gate_proj_q) = model.get_tensor(mtf_hash_name(&format!(
            "model.layers.{}.mlp.gate_proj.weight",
            l
        )))?;
        let (up_proj_p, up_proj_q) = model.get_tensor(mtf_hash_name(&format!(
            "model.layers.{}.mlp.up_proj.weight",
            l
        )))?;
        let (down_proj_p, down_proj_q) = model.get_tensor(mtf_hash_name(&format!(
            "model.layers.{}.mlp.down_proj.weight",
            l
        )))?;

        layers.push(ActiveLayer {
            input_layernorm: decode_weight_matrix(in_norm_p, in_norm_q),
            q_proj: decode_weight_matrix(q_proj_p, q_proj_q),
            q_bias,
            k_proj: decode_weight_matrix(k_proj_p, k_proj_q),
            k_bias,
            v_proj: decode_weight_matrix(v_proj_p, v_proj_q),
            v_bias,
            o_proj: decode_weight_matrix(o_proj_p, o_proj_q),
            post_attention_layernorm: decode_weight_matrix(post_norm_p, post_norm_q),
            gate_proj: decode_weight_matrix(gate_proj_p, gate_proj_q),
            up_proj: decode_weight_matrix(up_proj_p, up_proj_q),
            down_proj: decode_weight_matrix(down_proj_p, down_proj_q),
        });
    }

    let (norm_payload, norm_qtype) = model.get_tensor(mtf_hash_name("model.norm.weight"))?;
    let final_norm_weight = decode_weight_matrix(norm_payload, norm_qtype);

    let (embed_payload, embed_qtype) =
        model.get_tensor(mtf_hash_name("model.embed_tokens.weight"))?;
    let embed_tokens_f32 = decode_weight_matrix(embed_payload, embed_qtype);

    // lm_head fallback (tied embeddings)
    let lm_head_f32 = match model.get_tensor(mtf_hash_name("lm_head.weight")) {
        Ok((payload, qtype)) => {
            println!("[+] Loaded separate lm_head.weight");
            decode_weight_matrix(payload, qtype)
        }
        Err(_) => {
            println!("[WARN] lm_head.weight not found; using tied embeddings (embed_tokens) as output projection.");
            embed_tokens_f32.clone()
        }
    };

    println!(
        "[SUCCESS] Pre-decoded dynamic Model layers in {:?}",
        start_decode.elapsed()
    );

    println!("\n[Launch] Launching True CPU Inference Terminal (F32)");
    println!("-------------------------------------------------------");

    let tokenizer = DynamicTokenizer::new(model.get_metadata());

    // ---- Get BOS and EOS from config dynamically ----
    let bos_token_id = config.bos_token_id.unwrap_or(151643);
    let eos_token_id = config.eos_token_id.unwrap_or(151643);

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
        let mut gen_tokens = tokenizer.tokenize(prompt);
        if gen_tokens.is_empty() {
            continue;
        }

        // ---- Prepend BOS ----
        gen_tokens.insert(0, bos_token_id);
        println!("[*] Tokenized prompt + BOS: {:?}", gen_tokens);

        print!("Assistant ❯ ");
        stdout().flush()?;

        let max_new_tokens = 20;
        let temperature = 0.7;

        for _step in 0..max_new_tokens {
            let seq_len = gen_tokens.len();
            let mut x_seq = vec![0.0f32; seq_len * hidden_size];

            for i in 0..seq_len {
                let token_id = gen_tokens[i] as usize;
                let start = token_id * hidden_size;
                let end = start + hidden_size;
                let slice = if end <= embed_tokens_f32.len() {
                    &embed_tokens_f32[start..end]
                } else {
                    &embed_tokens_f32[0..hidden_size]
                };
                x_seq[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(slice);
            }

            // Forward pass
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
                        rms_norm_eps,
                    );
                    let mut q_i = vec![0.0f32; hidden_size];
                    let mut k_i = vec![0.0f32; kv_proj_size];
                    let mut v_i = vec![0.0f32; kv_proj_size];

                    // Check for biases dynamically
                    if let Some(bias) = &layer.q_bias {
                        matmul_vec_bias(
                            &mut q_i,
                            &layer.q_proj,
                            &x_norm,
                            bias,
                            hidden_size,
                            hidden_size,
                        );
                    } else {
                        matmul_vec(&mut q_i, &layer.q_proj, &x_norm, hidden_size, hidden_size);
                    }

                    if let Some(bias) = &layer.k_bias {
                        matmul_vec_bias(
                            &mut k_i,
                            &layer.k_proj,
                            &x_norm,
                            bias,
                            kv_proj_size,
                            hidden_size,
                        );
                    } else {
                        matmul_vec(&mut k_i, &layer.k_proj, &x_norm, kv_proj_size, hidden_size);
                    }

                    if let Some(bias) = &layer.v_bias {
                        matmul_vec_bias(
                            &mut v_i,
                            &layer.v_proj,
                            &x_norm,
                            bias,
                            kv_proj_size,
                            hidden_size,
                        );
                    } else {
                        matmul_vec(&mut v_i, &layer.v_proj, &x_norm, kv_proj_size, hidden_size);
                    }

                    // Apply RoPE with dynamic theta
                    apply_rope(
                        &mut q_i, &mut k_i, i, n_q_heads, n_kv_heads, head_dim, rope_theta,
                    );

                    q_seq[i * hidden_size..(i + 1) * hidden_size].copy_from_slice(&q_i);
                    k_seq[i * kv_proj_size..(i + 1) * kv_proj_size].copy_from_slice(&k_i);
                    v_seq[i * kv_proj_size..(i + 1) * kv_proj_size].copy_from_slice(&v_i);
                }

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

                for i in 0..seq_len {
                    rms_norm(
                        &x_post_attn[i * hidden_size..(i + 1) * hidden_size],
                        &layer.post_attention_layernorm,
                        &mut x_norm,
                        rms_norm_eps,
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

            let last_hidden = &x_seq[(seq_len - 1) * hidden_size..seq_len * hidden_size];
            let mut final_norm = vec![0.0f32; hidden_size];
            rms_norm(
                last_hidden,
                &final_norm_weight,
                &mut final_norm,
                rms_norm_eps,
            );

            // Compute logits
            let mut logits = vec![0.0f32; vocab_size];
            for vocab_id in 0..vocab_size {
                let start = vocab_id * hidden_size;
                let end = start + hidden_size;
                if end > lm_head_f32.len() {
                    continue;
                }
                let lm_row = &lm_head_f32[start..end];
                logits[vocab_id] = lm_row
                    .iter()
                    .zip(final_norm.iter())
                    .map(|(w, x)| w * x)
                    .sum();
            }

            // Repetition penalty
            let penalty = 1.1;
            for &token_id in &gen_tokens {
                let idx = token_id as usize;
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }

            // ---- Temperature sampling ----
            let mut rng = rand::thread_rng();
            let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = logits
                .iter()
                .map(|&x| ((x - max_logit) / temperature).exp())
                .sum();
            let mut probs = Vec::with_capacity(vocab_size);
            let mut cumulative = 0.0;
            for &x in &logits {
                let p = ((x - max_logit) / temperature).exp() / exp_sum;
                cumulative += p;
                probs.push(cumulative);
            }
            let sample: f32 = rng.gen();
            let mut predicted_token_id = 0;
            for (i, &c) in probs.iter().enumerate() {
                if sample <= c {
                    predicted_token_id = i as u32;
                    break;
                }
            }

            if predicted_token_id == eos_token_id {
                break;
            }

            gen_tokens.push(predicted_token_id);
            let word = tokenizer.decode_token(predicted_token_id).replace('Ġ', " ");
            print!("{}", word);
            stdout().flush()?;
        }

        let elapsed = start_time.elapsed();
        println!("\n-------------------------------------------------------");
        println!("  CPU Inference Latency: {:?}", elapsed);
        println!("-------------------------------------------------------");
    }

    Ok(())
}
