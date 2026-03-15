# PowerBench-Consumer

Benchmarking LLM inference and GRPO reasoning RL training on consumer GPUs (RTX 2050, 4GB VRAM).



---

## Key Results

### Inference (Qwen3-0.6B, batch size 1, input 32, output 32)

| Precision | Model Size (MB) | Mean Latency (ms) | Tokens/sec | WikiPPL | GSM8K |
|-----------|-----------------|-------------------|------------|---------|-------|
| FP16      | 1434.14         | 3061.50           | 10.45      | 19.89   | 6.0%  |
| INT8      | 3154.45         | 9642.51           | 3.32       | 19.93   | 11.0% |
| INT4      | 3154.45         | 3403.60           | 9.40       | 22.47   | 4.0%  |

**Finding:** INT8 is approximately 3x *slower* than FP16 on consumer hardware. This confirms that quantization benefits are not portable across hardware architectures (e.g., Jetson vs. RTX 2050). Language modeling (WikiPPL) remains robust, but reasoning accuracy at this scale is minimal.

### GRPO Training (Qwen3-0.6B + QLoRA rank-16)

| Metric | Value |
|--------|-------|
| Mean rollout time | 107,417 ms |
| Rollouts/minute | 2.2 |
| Peak VRAM | 911 MB |
| Mean power draw | 12.26 W |
| OOM count (5 steps) | 0 |
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
| 1     | 2/2             | 0.50       | 76.22 J     |
| 2     | 2/2             | 0.38       | 76.85 J     |
| 3     | 2/2             | 0.75       | 74.53 J     |
| Total |                 |            | 215.83 J / 1000 J budget |

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
python head_scorer.py --model Qwen/Qwen3-0.6B

# CALM composition
python calm_compose.py

# Energy-aware federated GRPO simulation
python fedgrpo_energy.py
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
