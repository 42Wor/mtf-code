use anyhow::{bail, Result};
use serde_json::Value;
use std::collections::HashMap;

pub fn validate_tensor_shapes(tensor_names: &[(String, Vec<usize>)], config: &Value) -> Result<()> {
    let model_type = config
        .get("model_type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    log::debug!("Validating shapes for model type: {}", model_type);

    let mut name_to_shape: HashMap<String, Vec<usize>> = HashMap::new();
    for (name, shape) in tensor_names {
        name_to_shape.insert(name.clone(), shape.clone());
    }

    let expected = match model_type {
        "llama" | "mistral" | "gemma" | "gpt2" => {
            get_expected_shapes_for_architecture(model_type, config)
        }
        _ => {
            log::warn!(
                "Unknown model type '{}' – skipping detailed shape validation",
                model_type
            );
            return Ok(());
        }
    };

    let mut errors = Vec::new();
    for (name, expected_shape) in expected {
        if let Some(actual_shape) = name_to_shape.get(&name) {
            if actual_shape != &expected_shape {
                errors.push(format!(
                    "{}: expected {:?}, got {:?}",
                    name, expected_shape, actual_shape
                ));
            }
        } else {
            log::debug!("Tensor {} not found in safetensors – optional?", name);
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        bail!("Shape validation failed:\n{}", errors.join("\n"));
    }
}

fn get_expected_shapes_for_architecture(arch: &str, config: &Value) -> HashMap<String, Vec<usize>> {
    let mut map = HashMap::new();

    let get_int = |key: &str| -> Option<usize> {
        config.get(key).and_then(|v| v.as_u64()).map(|v| v as usize)
    };

    let hidden_size = get_int("hidden_size").unwrap_or(4096);
    let num_heads = get_int("num_attention_heads").unwrap_or(32);
    let num_kv_heads = get_int("num_key_value_heads").unwrap_or(num_heads);
    let num_layers = get_int("num_hidden_layers").unwrap_or(32);
    let vocab_size = get_int("vocab_size").unwrap_or(32000);
    let head_dim = get_int("head_dim").unwrap_or(hidden_size / num_heads);
    let intermediate_size = get_int("intermediate_size").unwrap_or(hidden_size * 4);

    match arch {
        "llama" | "mistral" => {
            map.insert(
                "model.embed_tokens.weight".to_string(),
                vec![vocab_size, hidden_size],
            );
            map.insert("lm_head.weight".to_string(), vec![vocab_size, hidden_size]);
            for i in 0..num_layers {
                let prefix = format!("model.layers.{}.", i);
                map.insert(
                    format!("{}input_layernorm.weight", prefix),
                    vec![hidden_size],
                );
                map.insert(
                    format!("{}post_attention_layernorm.weight", prefix),
                    vec![hidden_size],
                );
                // Q, K, V, O projections
                map.insert(
                    format!("{}self_attn.q_proj.weight", prefix),
                    vec![hidden_size, hidden_size],
                );
                map.insert(
                    format!("{}self_attn.k_proj.weight", prefix),
                    vec![hidden_size, head_dim * num_kv_heads],
                );
                map.insert(
                    format!("{}self_attn.v_proj.weight", prefix),
                    vec![hidden_size, head_dim * num_kv_heads],
                );
                map.insert(
                    format!("{}self_attn.o_proj.weight", prefix),
                    vec![hidden_size, hidden_size],
                );
                // MLP
                map.insert(
                    format!("{}mlp.gate_proj.weight", prefix),
                    vec![hidden_size, intermediate_size],
                );
                map.insert(
                    format!("{}mlp.up_proj.weight", prefix),
                    vec![hidden_size, intermediate_size],
                );
                map.insert(
                    format!("{}mlp.down_proj.weight", prefix),
                    vec![intermediate_size, hidden_size],
                );
            }
        }

        "gpt2" => {
            map.insert("wte.weight".to_string(), vec![vocab_size, hidden_size]);
            map.insert("lm_head.weight".to_string(), vec![vocab_size, hidden_size]);
            for i in 0..num_layers {
                let prefix = format!("h.{}.", i);
                map.insert(format!("{}ln_1.weight", prefix), vec![hidden_size]);
                map.insert(format!("{}ln_2.weight", prefix), vec![hidden_size]);
                // GPT-2 combines QKV into one matrix: [hidden_size, 3*hidden_size]
                map.insert(
                    format!("{}attn.c_attn.weight", prefix),
                    vec![hidden_size, hidden_size * 3],
                );
                map.insert(
                    format!("{}attn.c_proj.weight", prefix),
                    vec![hidden_size, hidden_size],
                );
                map.insert(
                    format!("{}mlp.c_fc.weight", prefix),
                    vec![hidden_size, intermediate_size],
                );
                map.insert(
                    format!("{}mlp.c_proj.weight", prefix),
                    vec![intermediate_size, hidden_size],
                );
            }
        }
        "gemma" => {
            map.insert(
                "model.embed_tokens.weight".to_string(),
                vec![vocab_size, hidden_size],
            );
            map.insert("lm_head.weight".to_string(), vec![vocab_size, hidden_size]);
            for i in 0..num_layers {
                let prefix = format!("model.layers.{}.", i);
                map.insert(
                    format!("{}input_layernorm.weight", prefix),
                    vec![hidden_size],
                );
                map.insert(
                    format!("{}post_attention_layernorm.weight", prefix),
                    vec![hidden_size],
                );
                map.insert(
                    format!("{}self_attn.q_proj.weight", prefix),
                    vec![hidden_size, hidden_size],
                );
                map.insert(
                    format!("{}self_attn.k_proj.weight", prefix),
                    vec![hidden_size, head_dim * num_kv_heads],
                );
                map.insert(
                    format!("{}self_attn.v_proj.weight", prefix),
                    vec![hidden_size, head_dim * num_kv_heads],
                );
                map.insert(
                    format!("{}self_attn.o_proj.weight", prefix),
                    vec![hidden_size, hidden_size],
                );
                map.insert(
                    format!("{}mlp.gate_proj.weight", prefix),
                    vec![hidden_size, intermediate_size],
                );
                map.insert(
                    format!("{}mlp.up_proj.weight", prefix),
                    vec![hidden_size, intermediate_size],
                );
                map.insert(
                    format!("{}mlp.down_proj.weight", prefix),
                    vec![intermediate_size, hidden_size],
                );
            }
        }
        "qwen2" => {
            map.insert(
                "model.embed_tokens.weight".to_string(),
                vec![vocab_size, hidden_size],
            );
            map.insert("lm_head.weight".to_string(), vec![vocab_size, hidden_size]);
            for i in 0..num_layers {
                let prefix = format!("model.layers.{}.", i);
                map.insert(
                    format!("{}input_layernorm.weight", prefix),
                    vec![hidden_size],
                );
                map.insert(
                    format!("{}post_attention_layernorm.weight", prefix),
                    vec![hidden_size],
                );
                map.insert(
                    format!("{}self_attn.q_proj.weight", prefix),
                    vec![hidden_size, hidden_size],
                );
                map.insert(
                    format!("{}self_attn.k_proj.weight", prefix),
                    vec![hidden_size, head_dim * num_kv_heads],
                );
                map.insert(
                    format!("{}self_attn.v_proj.weight", prefix),
                    vec![hidden_size, head_dim * num_kv_heads],
                );
                map.insert(
                    format!("{}self_attn.o_proj.weight", prefix),
                    vec![hidden_size, hidden_size],
                );
                map.insert(
                    format!("{}mlp.gate_proj.weight", prefix),
                    vec![hidden_size, intermediate_size],
                );
                map.insert(
                    format!("{}mlp.up_proj.weight", prefix),
                    vec![hidden_size, intermediate_size],
                );
                map.insert(
                    format!("{}mlp.down_proj.weight", prefix),
                    vec![intermediate_size, hidden_size],
                );
            }
        }

        _ => {}
    }
    map
}
