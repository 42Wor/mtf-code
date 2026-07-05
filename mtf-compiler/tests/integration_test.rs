#[test]
fn test_compile_tiny_llama() {
    let input_dir = std::path::Path::new("tests/fixtures/tiny_llama");
    if !input_dir.join("model.safetensors").exists() {
        eprintln!(
            "Skipping integration test: no fixture found at {:?}",
            input_dir
        );
        return;
    }
    let output = std::env::temp_dir().join("test_output.mtf");
    let result = mtf_compiler::compiler::run_compile(input_dir, &output);
    assert!(result.is_ok());
    assert!(output.exists());
    std::fs::remove_file(output).unwrap();
}
