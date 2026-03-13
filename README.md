# PowerBench-Consumer

Benchmarking LLM inference and GRPO reasoning RL training on consumer GPUs (RTX 2050, 4GB VRAM).

Extends [DREAM:Lab's PAISE 2025 Jetson Orin AGX benchmark](https://dream-lab.in) to the sub-4GB consumer hardware regime.

---

## Key Results

### Inference (Qwen3-0.6B, batch size 1, input 32, output 32)

| Precision | Model Size (MB) | Mean Latency (ms) | Tokens/sec | Perplexity |
|-----------|-----------------|-------------------|------------|------------|
| FP16      | 1433.62         | 2407.20           | 13.29      | 14.80      |
| INT8      | 1013.62         | 10056.30          | 3.18       | 19.46      |
| INT4      | 803.62          | 3965.26           | 8.07       | 15.36      |

**Finding:** INT8 is approximately 3x *slower* than FP16 on consumer hardware, the opposite
of results on the Jetson Orin AGX. Consumer GPUs lack dedicated INT8 tensor cores, so
quantization overhead dominates rather than helping. Quantization benefits are not portable
across hardware architectures.

### GRPO Training (Qwen3-0.6B + QLoRA rank-16)

| Metric | Value |
|--------|-------|
| Mean rollout time | 159,132 ms |
| Rollouts/minute | 1.5 |
| Peak VRAM | 1,151 MB |
| Mean power draw | 14.3 W |
| OOM count (100 steps) | 0 |
| Long-tail frequency | 100% |

Every sequence exceeded the long-tail threshold (384 tokens), making consumer-hardware
GRPO the worst-case scenario for RL training efficiency. This motivates adaptive
speculative decoding approaches (see TLT, ASPLOS 2026) at the consumer scale.

---

## Files

| File | Description |
|------|-------------|
| `power_monitor.py` | Background thread GPU power sampling via nvidia-smi |
| `benchmark_inference.py` | Inference benchmark across FP16/INT8/INT4 |
| `benchmark_training.py` | GRPO training throughput and VRAM benchmark |
| `head_scorer.py` | TV-distance attention head scoring (DisCEdit analogue for transformers) |
| `calm_compose.py` | CALM-style cross-attention composition of two Qwen3-0.6B checkpoints |
| `fedgrpo_energy.py` | Energy-aware federated GRPO simulation (Flotilla-inspired) |

---

## Head Scoring Results (head_scorer.py)

448 attention heads scored (28 layers x 16 heads) by TV-distance between
reasoning prompts (math problems) and general text distributions.

**Top discriminative heads (reasoning-sensitive):**
- Layer 21 Head 1: TV = 0.9073
- Layer 21 Head 7: TV = 0.9073
- Layer 22 Head 12: TV = 0.8916

**Bottom heads (pruning candidates):**
- Layer 14 Head 0: TV = 0.2224
- Layer 2 Head 9: TV = 0.1794

Reasoning sensitivity is concentrated in later layers, consistent with reasoning
being a higher-level operation.

---

## CALM Composition (calm_compose.py)

Cross-attention bridges inserted at layers 4, 8, 12. Both base models frozen.
Trainable parameters: ~2M (W_q, W_k, W_v, W_o projections only).

Baseline perplexity (anchor = augmentor = base model): **25.39**

Next step: use GRPO-trained checkpoint as augmentor and evaluate on GSM8K.

**Note:** This implementation uses a simplified forward pass (all hidden states
collected upfront) rather than true layer-by-layer composition, to fit within
4GB VRAM. This is a known simplification of the original CALM paper.

---

## Energy-Aware Federated GRPO (fedgrpo_energy.py)

Simulation of Flotilla-style federated GRPO where clients are selected each round
within a joule budget, using per-client energy predictors calibrated from live
nvidia-smi power readings.

| Round | Clients selected | Avg reward | Energy used |
|-------|-----------------|------------|-------------|
| 1     | 2/2             | 0.63       | 79.94 J     |
| 2     | 2/2             | 0.50       | 89.21 J     |
| 3     | 2/2             | 0.13       | 84.41 J     |
| Total |                 |            | 253.57 J / 1000 J budget |

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Usage

```bash
# GPU power monitor test
python power_monitor.py

# Inference benchmark
python benchmark_inference.py --model Qwen/Qwen3-0.6B

# Training benchmark
python benchmark_training.py --model Qwen/Qwen3-0.6B --steps 10

# Head scoring
python head_scorer.py --model_path Qwen/Qwen3-0.6B

# CALM composition
python calm_compose.py
```

---

## Hardware

- GPU: NVIDIA RTX 2050 (4GB VRAM)
- RAM: 16GB
- OS: Windows 11 / WSL2
- CUDA: 11.8

## References

- Simmhan et al., "LLM Inference on Edge Accelerators", PAISE 2025
- Simmhan et al., "Flotilla: Scalable Federated Learning", JPDC 2025
- Bhattacharyya et al., "DisCEdit", NeurIPS 2024
- Bansal, Talukdar et al., "CALM", ICLR 2024
- Han et al., "TLT: Taming the Long Tail", ASPLOS 2026
