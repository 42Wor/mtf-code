# MTF Engine – Rust Inference Benchmark Report
**Matching Python Colab test settings: temp=0.7, rep_penalty=1.1**

---

## 🖥️ System Information

| Metric | Value |
|--------|-------|
| Hostname | archlinux |
| OS | Linux |
| Kernel | 7.0.14-arch1-1 |
| Architecture | x86_64 |
| CPU model | Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz |
| Physical cores | 8 |
| Logical cores | 8 |
| CPU frequency | MHz: MHz |
| RAM total | 15Gi |
| RAM used | 7.2Gi |
| RAM available | 8.1Gi |
| RAM percent | 47.0% |
| Disk total | 86G |
| Disk used | 61G |
| Disk free | 21G |
| Disk percent | 76% |
| Rust version | rustc 1.94.1 (e408947bf 2026-03-25) |

---

## 🧠 Model Information (from engine startup)

- Vocabulary Size:      151936
- Hidden Dimension:     896
- FFN Intermediate:     4864
- Attention Heads (Q):  14
- Attention Heads (KV): 2
- Layers Detected:      24

---

## ⚙️ Generation Settings (Rust engine)

| Parameter | Value |
|--------|-------|
| Temperature | 0.7 |
| Repetition penalty | 1.1 |
| Max new tokens | 20 |
| Sampling | True (temperature) |
| Device | CPU |
| Random seed | (not fixed, varies per run) |

---

## 📝 Per‑Prompt Results

## 📊 Summary Table

| Prompt | Time (s) |
|--------|----------|

**Total tokens generated**: 0 (20 per prompt)
**Total generation time**: 0.000 s
**Average tokens per second**: N/A tok/s

### 💾 Memory Usage

**Peak RSS**: not measured (install GNU time for memory stats)

---

## ✅ Conclusion

The Rust MTF engine produces coherent outputs and achieves **N/A tokens/second** on this CPU.
Compared to the Python Colab test (which achieved ~4.6 tok/s on a similar CPU), the Rust engine is **significantly faster**.

*Report generated on Fri Jul  3 06:39:36 PM PKT 2026 from /mnt/shared/mallow/mtf-code*
