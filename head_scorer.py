"""
head_scorer.py
==============
TV-distance attention head scoring for Qwen3-0.6B.

Based on the Discriminative Components framework from:
  DisCEdit (Bhattacharyya lab, NeurIPS 2024)
  Original: CNN filter scoring via TV-distance between class activation distributions.
  This file: direct analogue for transformer attention heads.

Core idea:
  Feed two prompt sets (reasoning vs. general text) through the model.
  For each attention head in each layer, collect its output activations.
  Compute Total Variation distance between the two distributions.
  High TV = head responds differently to reasoning vs. general = "discriminative" = important.
  Low TV = head doesn't care what type of input = prunable.

Usage:
  scorer = HeadScorer(model, tokenizer, device)
  scores = scorer.score_all_heads(reasoning_prompts, general_prompts)
  pruner = HeadPruner(model)
  pruner.prune_bottom_k_percent(scores, k=20)
  pruner.evaluate(eval_dataloader)
"""

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class HeadScore:
    layer_idx: int
    head_idx: int
    tv_distance: float  # higher = more discriminative = keep
    mean_activation_reasoning: float
    mean_activation_general: float


# ---------------------------------------------------------------------------
# Module 1: Activation Collector
# Hook-based collector — attaches to each attention layer and records
# the per-head output activations for a batch of inputs.
# ---------------------------------------------------------------------------


class ActivationCollector:
    """
    Registers forward hooks on every attention layer of the model.
    After a forward pass, self.activations[layer_idx] contains
    a tensor of shape (batch, seq_len, num_heads, head_dim).
    """

    def __init__(self, model):
        self.model = model
        self.activations = {}  # {layer_idx: tensor}
        self._hooks = []

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # output is a tuple — the tensor we want is output[0], shape (batch, seq_len, hidden_size).
            tensor = output[0]
            batch, seq_len, hidden_size = tensor.shape
            num_heads = self.model.config.num_attention_heads
            head_dim = hidden_size // num_heads
            # Reshape it to (batch, seq_len, num_heads, head_dim)
            # Store the reshaped tensor in self.activations[layer_idx].
            # Use .detach().cpu() before storing to avoid holding GPU memory.
            self.activations[layer_idx] = tensor.view(batch, seq_len, num_heads, head_dim).detach().cpu()
        return hook

    def register(self):
        """
        Attach hooks to every attention module in the model.
        Qwen3 attention modules are at: model.model.layers[i].self_attn
        Iterate model.model.layers, find .self_attn, register hook.
        """
        for i, layer in enumerate(self.model.model.layers):
            handle = layer.self_attn.register_forward_hook(self._make_hook(i))
            self._hooks.append(handle)

    def clear(self):
        """Remove all hooks and reset stored activations."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.activations = {}


# ---------------------------------------------------------------------------
# Module 2: TV Distance Calculator
# Given two sets of activations (reasoning vs general),
# computes per-head Total Variation distance.
# ---------------------------------------------------------------------------


class TVDistanceCalculator:
    """
    TV distance between two empirical distributions P and Q:
      TV(P, Q) = 0.5 * sum(|P(x) - Q(x)|)
    We estimate this by binning activation values into a histogram.
    """

    def __init__(self, n_bins: int = 50):
        self.n_bins = n_bins

    def _to_histogram(self, activations: np.ndarray) -> np.ndarray:
        """
        activations: 1D array of scalar activation values.
        Returns: normalised histogram of shape (n_bins,).
        """
        # Use np.histogram with self.n_bins bins and range=(activations.min(), activations.max()).
        counts, _ = np.histogram(activations, bins=self.n_bins, range=(activations.min(), activations.max()))
        # Convert to float and add 1e-10 to avoid division by zero.
        counts = counts.astype(float) + 1e-10
        # Divide by the sum so it sums to 1.
        return counts / counts.sum()

    def compute(self, dist_a: np.ndarray, dist_b: np.ndarray) -> float:
        """
        dist_a, dist_b: 1D arrays of activation values from two prompt sets.
        Returns scalar TV distance in [0, 1].
        """
        # Call self._to_histogram on dist_a and dist_b separately.
        pa = self._to_histogram(dist_a)
        pb = self._to_histogram(dist_b)
        # TV distance = 0.5 * np.sum(np.abs(pa - pb)).
        return float(0.5 * np.sum(np.abs(pa - pb)))

    def per_head_tv(
        self,
        acts_reasoning: torch.Tensor,  # (n_prompts, seq_len, num_heads, head_dim)
        acts_general: torch.Tensor,  # (n_prompts, seq_len, num_heads, head_dim)
    ) -> np.ndarray:
        """
        Compute TV distance for every head independently.
        Returns array of shape (num_heads,).

        Strategy: for each head, take the mean activation across head_dim
        to get a scalar per (prompt, token), flatten to 1D, then compute TV.
        """
        # Step 1: take mean over the last dimension (head_dim) to get shape (n_prompts, seq_len, num_heads).
        r_mean = acts_reasoning.mean(dim=-1)
        g_mean = acts_general.mean(dim=-1)

        # Step 2: convert both to numpy with .detach().cpu().numpy().
        r_np = r_mean.detach().cpu().numpy()
        g_np = g_mean.detach().cpu().numpy()

        # Step 3: get num_heads from acts_reasoning.shape[2].
        num_heads = r_np.shape[2]
        tv_scores = []

        # Step 4: for each head index h from 0 to num_heads-1:
        for h in range(num_heads):
            # flatten acts_reasoning[:, :, h] to 1D
            reasoning_flat = r_np[:, :, h].flatten()
            # flatten acts_general[:, :, h] to 1D
            general_flat = g_np[:, :, h].flatten()
            # call self.compute(reasoning_flat, general_flat)
            tv = self.compute(reasoning_flat, general_flat)
            tv_scores.append(tv)

        # Step 5: return np.array(tv_scores) of shape (num_heads,).
        return np.array(tv_scores)


# ---------------------------------------------------------------------------
# Module 3: HeadScorer — orchestrates collection + scoring
# ---------------------------------------------------------------------------


class HeadScorer:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.collector = ActivationCollector(model)
        self.calculator = TVDistanceCalculator(n_bins=50)

    def _encode_prompts(self, prompts: list[str]) -> dict:
        """Tokenize a list of prompts, return input_ids on self.device."""
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

    def _collect_activations(self, prompts: list[str]) -> dict:
        """
        Run a forward pass over prompts and return collected activations.
        Returns: {layer_idx: tensor of shape (batch, seq, num_heads, head_dim)}
        """
        self.collector.register()
        inputs = self._encode_prompts(prompts)
        with torch.no_grad():
            self.model(**inputs)
        activations = dict(self.collector.activations)  # copy before clearing
        self.collector.clear()
        return activations

    def score_all_heads(
        self,
        reasoning_prompts: list[str],
        general_prompts: list[str],
    ) -> list[HeadScore]:
        """
        Main entry point. Returns a list of HeadScore objects,
        one per (layer, head), sorted by tv_distance descending
        (most discriminative first).
        """
        print("Collecting activations for reasoning prompts...")
        acts_r = self._collect_activations(reasoning_prompts)

        print("Collecting activations for general prompts...")
        acts_g = self._collect_activations(general_prompts)

        scores = []
        num_layers = len(acts_r)

        for layer_idx in range(num_layers):
            ar = acts_r[layer_idx]  # (batch, seq, heads, head_dim)
            ag = acts_g[layer_idx]

            tv_per_head = self.calculator.per_head_tv(ar, ag)

            for head_idx, tv in enumerate(tv_per_head):
                scores.append(
                    HeadScore(
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        tv_distance=float(tv),
                        mean_activation_reasoning=float(ar[:, :, head_idx, :].mean()),
                        mean_activation_general=float(ag[:, :, head_idx, :].mean()),
                    )
                )

        scores.sort(key=lambda s: s.tv_distance, reverse=True)
        return scores

    def save_scores(self, scores: list[HeadScore], path: str):
        with open(path, "w") as f:
            json.dump([s.__dict__ for s in scores], f, indent=2)
        print(f"Saved {len(scores)} head scores to {path}")


# ---------------------------------------------------------------------------
# Module 4: HeadPruner — zeros out low-scoring heads
# ---------------------------------------------------------------------------


class HeadPruner:
    def __init__(self, model):
        self.model = model
        self._pruned_heads = []  # list of (layer_idx, head_idx)

    def prune_bottom_k_percent(
        self,
        scores: list[HeadScore],
        k: float,  # e.g. 20 means prune bottom 20% of heads
    ):
        """
        Zero out the output projection weights corresponding to the
        bottom k% of heads by TV-distance score.

        For Qwen3, the output projection is at:
            model.model.layers[layer_idx].self_attn.o_proj.weight
        Shape: (hidden_size, hidden_size)
        Each head occupies a contiguous block of head_dim columns.
        Zeroing those columns effectively kills that head's contribution.
        """
        n_prune = int(len(scores) * k / 100)
        to_prune = scores[-n_prune:]  # lowest TV scores = least discriminative

        print(f"Pruning {n_prune} heads ({k}% of {len(scores)} total)")

        for score in to_prune:
            o_proj = self.model.model.layers[score.layer_idx].self_attn.o_proj
            num_heads = self.model.config.num_attention_heads
            head_dim = o_proj.weight.shape[1] // num_heads
            col_start = score.head_idx * head_dim
            with torch.no_grad():
                o_proj.weight[:, col_start:col_start+head_dim] = 0.0
            self._pruned_heads.append((score.layer_idx, score.head_idx))

    def evaluate(self, eval_dataloader, device="cuda") -> dict:
        """
        Compute perplexity on eval_dataloader after pruning.
        Returns {"perplexity": float, "n_pruned_heads": int}
        """
        self.model.eval()
        total_loss = 0.0
        total_steps = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                total_steps += 1
        
        mean_loss = total_loss / total_steps if total_steps > 0 else 0.0
        return {
            "perplexity": float(torch.exp(torch.tensor(mean_loss))),
            "n_pruned_heads": len(self._pruned_heads)
        }

    def pruning_curve(
        self,
        scores: list[HeadScore],
        eval_dataloader,
        k_values: list[float] = [0, 10, 20, 30, 40, 50],
    ) -> list[dict]:
        """
        Run evaluate() at each k in k_values and return results list.
        This is the table that goes in the Bhattacharyya email.
        """
        results = []
        for k in k_values:
            # reload clean model weights before each pruning level
            # (caller must handle reloading — just prune and evaluate here)
            self.prune_bottom_k_percent(scores, k)
            metrics = self.evaluate(eval_dataloader)
            metrics["k_percent"] = k
            results.append(metrics)
            print(f"k={k}%: perplexity={metrics['perplexity']:.3f}")
        return results


# ---------------------------------------------------------------------------
# Prompt sets — paste your own or extend these
# ---------------------------------------------------------------------------

REASONING_PROMPTS = [
    "Solve step by step: If 3x + 7 = 22, what is x?",
    "A train travels 60km/h for 2 hours then 80km/h for 3 hours. Total distance?",
    "If all bloops are razzles and all razzles are lazzles, are all bloops lazzles?",
    "What is 15% of 240?",
    "A rectangle has perimeter 36cm and length 10cm. Find the width.",
    "If log2(x) = 5, what is x?",
    "John has 3 times as many apples as Mary. Together they have 48. How many does John have?",
    "What is the derivative of x^3 + 2x^2 - 5x + 1?",
    "A bag has 4 red and 6 blue balls. Probability of drawing red?",
    "Simplify: (x^2 - 4) / (x - 2)",
]

GENERAL_PROMPTS = [
    "The capital of France is",
    "In the 19th century, the industrial revolution",
    "Photosynthesis is the process by which plants",
    "The human heart has four",
    "Shakespeare wrote his plays during the",
    "Water boils at 100 degrees",
    "The Amazon rainforest is located in",
    "Mount Everest is the tallest",
    "The theory of relativity was proposed by",
    "The Great Wall of China was built to",
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--k_values", nargs="+", type=float, default=[0, 10, 20, 30])
    parser.add_argument("--output", default="head_scores.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    scorer = HeadScorer(model, tokenizer, device)
    scores = scorer.score_all_heads(REASONING_PROMPTS, GENERAL_PROMPTS)
    scorer.save_scores(scores, args.output)

    print(f"\nTop 10 most discriminative heads:")
    for s in scores[:10]:
        print(f"  Layer {s.layer_idx:2d} Head {s.head_idx:2d}  TV={s.tv_distance:.4f}")

    print(f"\nBottom 10 (pruning candidates):")
    for s in scores[-10:]:
        print(f"  Layer {s.layer_idx:2d} Head {s.head_idx:2d}  TV={s.tv_distance:.4f}")
