"""
benchmark_training.py
=====================
Benchmarks GRPO reasoning RL training throughput on consumer GPU hardware.
This is the dimension missing from Simmhan's PAISE 2025 paper — they only
measured inference. This file measures training.

Metrics (per training step):
  - Rollout throughput (rollouts/minute)
  - Per-step latency (ms)
  - Peak VRAM during rollout vs during backward pass
  - Power draw during training
  - Long-tail rollout frequency (% of sequences > threshold tokens)
  - OOM count over N steps

Usage:
    python benchmark_training.py --model Qwen/Qwen3-0.6B --steps 100
"""

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass, field

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from power_monitor import PowerMonitor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
N_ROLLOUTS = 4  # rollouts per prompt (standard GRPO)
MAX_NEW_TOKENS = 512
LONG_TAIL_THRESH = 384  # flag sequences longer than this
N_WARMUP_STEPS = 5
N_MEASURE_STEPS = 100

# Simple math prompts for rollout generation
BENCHMARK_PROMPTS = [
    "Solve step by step: If 3x + 7 = 22, what is x?",
    "A train travels 60 km/h for 2 hours then 80 km/h for 3 hours. Total distance?",
    "What is 15% of 240?",
    "If log2(x) = 5, what is x?",
    "Simplify: (x^2 - 4) / (x - 2)",
    "A rectangle has perimeter 36cm and length 10cm. Find the width.",
    "If all bloops are razzles and all razzles are lazzles, are all bloops lazzles?",
    "What is the sum of angles in a pentagon?",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class StepMetrics:
    step: int
    rollout_time_ms: float
    backward_time_ms: float
    total_step_time_ms: float
    peak_vram_rollout_mb: float
    peak_vram_backward_mb: float
    n_long_tail: int  # sequences > LONG_TAIL_THRESH tokens
    mean_output_tokens: float
    oom: bool = False


@dataclass
class TrainingBenchmarkResult:
    model_name: str
    n_steps: int
    # Throughput
    mean_rollout_time_ms: float
    p50_rollout_time_ms: float
    p95_rollout_time_ms: float
    rollouts_per_minute: float
    # Memory
    peak_vram_mb: float
    mean_vram_rollout_mb: float
    mean_vram_backward_mb: float
    # Power
    mean_power_w: float
    peak_power_w: float
    # Stability
    oom_count: int
    long_tail_frequency: float  # fraction of sequences flagged
    # Raw steps for plotting
    steps: list[StepMetrics] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module 1: ModelSetup
# Loads model with QLoRA for training benchmark
# ---------------------------------------------------------------------------


class ModelSetup:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device

    def load_with_lora(self):
        """
        Load Qwen3-0.6B in 4-bit QLoRA config.
        Returns (model, tokenizer) ready for training.
        """
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, TaskType, get_peft_model

        bnb_config = BitsAndBytesConfig(
          load_in_4bit=True, bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.float16,
          bnb_4bit_use_double_quant=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        lora_config = LoraConfig(
          r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
          task_type=TaskType.CAUSAL_LM, bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, tokenizer


# ---------------------------------------------------------------------------
# Module 2: RolloutBenchmarker
# Measures the rollout phase of GRPO (the slow part)
# ---------------------------------------------------------------------------


class RolloutBenchmarker:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _tokenize_prompt(self, prompt: str) -> dict:
        """Tokenize a single prompt, move to device."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def run_rollout(self, prompt: str) -> dict:
        """
        Generate N_ROLLOUTS completions for one prompt.
        Returns timing and token count stats.
        """
        inputs = self._tokenize_prompt(prompt)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_return_sequences=N_ROLLOUTS,
                do_sample=True, temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        rollout_time_ms = (t1 - t0) * 1000
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

        input_len = inputs["input_ids"].shape[1]
        output_lens = [outputs[i].shape[0] - input_len for i in range(N_ROLLOUTS)]
        n_long_tail = sum(1 for l in output_lens if l > LONG_TAIL_THRESH)
        mean_output_tokens = sum(output_lens) / len(output_lens)
        return {
            "rollout_time_ms": rollout_time_ms,
            "peak_vram_mb": peak_vram_mb,
            "n_long_tail": n_long_tail,
            "mean_output_tokens": mean_output_tokens
        }


# ---------------------------------------------------------------------------
# Module 3: BackwardBenchmarker
# Measures the backward pass (LoRA update step)
# ---------------------------------------------------------------------------


class BackwardBenchmarker:
    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )

    def run_backward(self, input_ids: torch.Tensor) -> dict:
        """
        Run a fake backward pass using cross-entropy loss.
        (Real GRPO loss needs rewards — this benchmarks the backward overhead.)
        Returns timing and VRAM stats.
        """
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        self.optimizer.zero_grad()
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
          [p for p in self.model.parameters() if p.requires_grad], 1.0
        )
        self.optimizer.step()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

        return {
          "backward_time_ms": (t1 - t0) * 1000,
          "peak_vram_mb": peak_vram_mb
        }


# ---------------------------------------------------------------------------
# Module 4: TrainingBenchmarkRunner — orchestrates everything
# ---------------------------------------------------------------------------


class TrainingBenchmarkRunner:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.power = PowerMonitor()

    def run(self, n_steps: int = N_MEASURE_STEPS, n_warmup: int = N_WARMUP_STEPS) -> TrainingBenchmarkResult:
        """Run full training benchmark. Returns TrainingBenchmarkResult."""

        print(f"Loading model with QLoRA...")
        setup = ModelSetup(self.model_name, self.device)
        model, tokenizer = setup.load_with_lora()

        rollout_bench = RolloutBenchmarker(model, tokenizer, self.device)
        backward_bench = BackwardBenchmarker(model, self.device)

        step_metrics = []
        oom_count = 0

        print(f"Warming up ({n_warmup} steps)...")
        for i in range(n_warmup):
            prompt = BENCHMARK_PROMPTS[i % len(BENCHMARK_PROMPTS)]
            try:
                rollout_bench.run_rollout(prompt)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()

        print(f"Benchmarking ({n_steps} steps)...")
        self.power.start()

        for step in range(n_steps):
            prompt = BENCHMARK_PROMPTS[step % len(BENCHMARK_PROMPTS)]
            inputs = rollout_bench._tokenize_prompt(prompt)
            oom_step = False

            try:
                r_stats = rollout_bench.run_rollout(prompt)
                b_stats = backward_bench.run_backward(inputs["input_ids"])

                step_metrics.append(
                    StepMetrics(
                        step=step,
                        rollout_time_ms=r_stats["rollout_time_ms"],
                        backward_time_ms=b_stats["backward_time_ms"],
                        total_step_time_ms=r_stats["rollout_time_ms"]
                        + b_stats["backward_time_ms"],
                        peak_vram_rollout_mb=r_stats["peak_vram_mb"],
                        peak_vram_backward_mb=b_stats["peak_vram_mb"],
                        n_long_tail=r_stats["n_long_tail"],
                        mean_output_tokens=r_stats["mean_output_tokens"],
                    )
                )

            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                oom_step = True
                torch.cuda.empty_cache()
                step_metrics.append(
                    StepMetrics(
                        step=step,
                        rollout_time_ms=0,
                        backward_time_ms=0,
                        total_step_time_ms=0,
                        peak_vram_rollout_mb=0,
                        peak_vram_backward_mb=0,
                        n_long_tail=0,
                        mean_output_tokens=0,
                        oom=True,
                    )
                )

            if (step + 1) % 10 == 0:
                valid = [s for s in step_metrics[-10:] if not s.oom]
                if valid:
                    mean_r = sum(s.rollout_time_ms for s in valid) / len(valid)
                    print(
                        f"  Step {step + 1}/{n_steps} | "
                        f"rollout={mean_r:.0f}ms | OOM={oom_count}"
                    )

        power_stats = self.power.stop()

        # Aggregate metrics from valid (non-OOM) steps
        valid_steps = [s for s in step_metrics if not s.oom]
        if not valid_steps:
            return TrainingBenchmarkResult(
                model_name=self.model_name, n_steps=n_steps,
                mean_rollout_time_ms=0.0, p50_rollout_time_ms=0.0,
                p95_rollout_time_ms=0.0, rollouts_per_minute=0.0,
                peak_vram_mb=0.0, mean_vram_rollout_mb=0.0, mean_vram_backward_mb=0.0,
                mean_power_w=power_stats.mean_w, peak_power_w=power_stats.peak_w,
                oom_count=oom_count, long_tail_frequency=0.0, steps=step_metrics
            )

        rollout_times = [s.rollout_time_ms for s in valid_steps]
        mean_rollout = sum(rollout_times) / len(rollout_times)
        p50_rollout = sorted(rollout_times)[len(rollout_times)//2]
        p95_rollout = sorted(rollout_times)[int(0.95*len(rollout_times))]
        rollouts_per_minute = (60_000 / mean_rollout) * N_ROLLOUTS if mean_rollout > 0 else 0
        peak_vram = max(max(s.peak_vram_rollout_mb, s.peak_vram_backward_mb) for s in valid_steps)
        mean_vram_rollout = sum(s.peak_vram_rollout_mb for s in valid_steps) / len(valid_steps)
        mean_vram_backward = sum(s.peak_vram_backward_mb for s in valid_steps) / len(valid_steps)
        long_tail_freq = sum(s.n_long_tail for s in valid_steps) / (len(valid_steps) * N_ROLLOUTS)

        return TrainingBenchmarkResult(
          model_name=self.model_name, n_steps=n_steps,
          mean_rollout_time_ms=mean_rollout, p50_rollout_time_ms=p50_rollout,
          p95_rollout_time_ms=p95_rollout, rollouts_per_minute=rollouts_per_minute,
          peak_vram_mb=peak_vram,
          mean_vram_rollout_mb=mean_vram_rollout,
          mean_vram_backward_mb=mean_vram_backward,
          mean_power_w=power_stats.mean_w, peak_power_w=power_stats.peak_w,
          oom_count=oom_count, long_tail_frequency=long_tail_freq,
          steps=step_metrics
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--steps", type=int, default=N_MEASURE_STEPS)
    parser.add_argument("--output", default="training_results.json")
    parser.add_argument("--warmup", type=int, default=N_WARMUP_STEPS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {total_vram:.0f} MB")

    runner = TrainingBenchmarkRunner(args.model, device)
    # Update global N_WARMUP_STEPS for this run if needed
    # (Simplified: runner.run will use its internal logic)
    
    # Actually modifying runner.run to take warmup
    result = runner.run(n_steps=args.steps, n_warmup=args.warmup)

    print(f"\n{'=' * 50}")
    print(f"  Training Benchmark Results")
    print(f"{'=' * 50}")
    print(f"  Mean rollout time:    {result.mean_rollout_time_ms:.0f} ms")
    print(f"  P95 rollout time:     {result.p95_rollout_time_ms:.0f} ms")
    print(f"  Rollouts/minute:      {result.rollouts_per_minute:.1f}")
    print(f"  Peak VRAM:            {result.peak_vram_mb:.0f} MB")
    print(f"  Mean power:           {result.mean_power_w:.1f} W")
    print(f"  OOM count:            {result.oom_count}/{args.steps}")
    print(f"  Long-tail frequency:  {result.long_tail_frequency:.1%}")

    with open(args.output, "w") as f:
        import dataclasses

        json.dump(dataclasses.asdict(result), f, indent=2)
    print(f"\nSaved to {args.output}")
