"""
benchmark_inference.py
======================
Benchmarks LLM inference latency, WikiText-2 perplexity, and GSM8K accuracy.
"""

import argparse
import gc
import time
import torch
import re
from dataclasses import dataclass, field
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

@dataclass
class InferenceResult:
    model_name: str
    precision: str
    batch_size: int
    input_length: int
    mean_latency_ms: float
    tokens_per_second: float
    vram_mb: float
    perplexity: float
    gsm8k_accuracy: float = 0.0

class ModelLoader:
    @staticmethod
    def load(model_name: str, precision: str = "fp16", device: str = "cuda"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        quantization_config = None
        if precision == "int8":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif precision == "int4":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_quant_type="nf4", 
                bnb_4bit_compute_dtype=torch.float16
            )
            
        if quantization_config:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                quantization_config=quantization_config, 
                device_map=device
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16, 
                device_map=device
            )
            
        model.eval()
        return model, tokenizer

    @staticmethod
    def model_size_mb(model):
        return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)

    @staticmethod
    def unload(model):
        del model
        gc.collect()
        torch.cuda.empty_cache()

class LatencyMeasurer:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _make_inputs(self, batch_size: int, input_length: int):
        prompt = "Solve the following math problem step by step: "
        inputs = self.tokenizer(
            prompt, 
            max_length=input_length, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        input_ids = inputs["input_ids"].repeat(batch_size, 1).to(self.device)
        attention_mask = inputs["attention_mask"].repeat(batch_size, 1).to(self.device)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def measure(self, batch_size: int, input_length: int, output_length: int, n_warmup: int = 3, n_measure: int = 10):
        inputs = self._make_inputs(batch_size, input_length)
        generate_kwargs = {"max_new_tokens": output_length, "do_sample": True, "min_new_tokens": output_length}
        
        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                self.model.generate(**inputs, **generate_kwargs)
        
        latencies = []
        for _ in range(n_measure):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                self.model.generate(**inputs, **generate_kwargs)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
            
        mean_lat = sum(latencies) / len(latencies)
        return {
            "mean_latency_ms": mean_lat,
            "tokens_per_second": batch_size * output_length / (mean_lat / 1000)
        }

class PerplexityMeasurer:
    """
    WikiText-2 perplexity with sliding window.
    Same methodology used to evaluate GPT-2, LLaMA, Qwen, etc.
    Lower is better. Directly comparable across papers.
    """

    def __init__(self, device="cuda", stride=512, max_length=1024):
        self.device = device
        self.stride = stride
        self.max_length = max_length

    def measure(self, model, tokenizer) -> float:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1",
                               split="test", trust_remote_code=True)
        text = "\n\n".join(dataset["text"])
        encodings = tokenizer(text, return_tensors="pt")

        seq_len = encodings.input_ids.size(1)
        nlls = []
        prev_end = 0

        for begin in range(0, seq_len, self.stride):
            if (begin // self.stride) % 50 == 0:
                print(f"    PPL Progress: {begin/seq_len:.1%}")
            end = min(begin + self.max_length, seq_len)
            target_len = end - prev_end
            input_ids = encodings.input_ids[:, begin:end].to(self.device)

            target_ids = input_ids.clone()
            target_ids[:, :-target_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                nlls.append(outputs.loss * target_len)

            prev_end = end
            if end == seq_len:
                break

        ppl = float(torch.exp(torch.stack(nlls).sum() / seq_len))
        return ppl

class GSM8KEvaluator:
    """
    GSM8K pass@1 accuracy. Standard reasoning benchmark.
    Samples n_problems from the test set, generates greedy responses,
    extracts the final number, checks against ground truth.
    """

    def __init__(self, device="cuda", n_problems=100):
        self.device = device
        self.n_problems = n_problems

    def _extract_number(self, text: str) -> str:
        # Matches integers and decimals, optional negative sign
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return numbers[-1] if numbers else None

    def measure(self, model, tokenizer) -> dict:
        dataset = load_dataset("gsm8k", "main",
                               split="test", trust_remote_code=True)
        dataset = dataset.select(range(self.n_problems))

        correct = 0
        for i, item in enumerate(dataset):
            print(f"    GSM8K Progress: {i}/{self.n_problems}")
            prompt = f"Question: {item['question']}\nAnswer:"
            inputs = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=256
            ).to(self.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )

            pred = self._extract_number(response)
            gold = self._extract_number(item["answer"].split("####")[-1].strip())

            print(f"      Q: {item['question'][:50]}...")
            print(f"      Pred: {pred} | Gold: {gold}")

            if pred == gold:
                correct += 1

        return {
            "accuracy": correct / self.n_problems,
            "n_correct": correct,
            "n_total": self.n_problems
        }

class BenchmarkRunner:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.results: List[InferenceResult] = []

    def run_single(self, precision: str, batch_size: int = 1, input_length: int = 32, output_length: int = 32):
        print(f"\nBenchmarking {precision} (BS={batch_size}, In={input_length})...")
        try:
            model, tokenizer = ModelLoader.load(self.model_name, precision=precision, device=self.device)
            vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2) if self.device == "cuda" else 0.0
            
            latency_measurer = LatencyMeasurer(model, tokenizer, self.device)
            lat_stats = latency_measurer.measure(batch_size=batch_size, input_length=input_length, output_length=output_length)
            
            print(f"  Measuring Perplexity...")
            ppl_measurer = PerplexityMeasurer(device=self.device)
            ppl = ppl_measurer.measure(model, tokenizer)

            print(f"  Measuring GSM8K (100 problems)...")
            gsm8k_evaluator = GSM8KEvaluator(device=self.device, n_problems=100)
            gsm8k_results = gsm8k_evaluator.measure(model, tokenizer)
            gsm8k_acc = gsm8k_results["accuracy"]

            res = InferenceResult(
                model_name=self.model_name,
                precision=precision,
                batch_size=batch_size,
                input_length=input_length,
                mean_latency_ms=lat_stats["mean_latency_ms"],
                tokens_per_second=lat_stats["tokens_per_second"],
                vram_mb=vram_mb,
                perplexity=ppl,
                gsm8k_accuracy=gsm8k_acc,
            )
            self.results.append(res)
            ModelLoader.unload(model)
            return res
        except Exception as e:
            print(f"Failed to benchmark {precision}: {e}")
            return None

    def print_summary(self, output_file: str = "inference_results.json"):
        import json
        print(f"\n{'='*100}")
        print(f"{'Inference Benchmark Summary':^100}")
        print(f"{'='*100}")
        print(f"{'Model':<20} {'Quant':<6} {'BS':<4} "
              f"{'Lat(ms)':<10} {'Tok/s':<10} {'VRAM(MB)':<10} "
              f"{'WikiPPL':<10} {'GSM8K':<8}")
        print("-" * 100)
        serialized_results = []
        for r in self.results:
            print(f"{r.model_name[:20]:<20} {r.precision:<6} {r.batch_size:<4} "
                  f"{r.mean_latency_ms:<10.2f} {r.tokens_per_second:<10.2f} {r.vram_mb:<10.2f} "
                  f"{r.perplexity:<10.2f} {r.gsm8k_accuracy:<8.2%}")
            serialized_results.append(r.__dict__)
        print(f"{'='*100}\n")
        
        with open(output_file, "w") as f:
            json.dump(serialized_results, f, indent=2)
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output", default="inference_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runner = BenchmarkRunner(args.model, device)
    
    for prec in ["fp16", "int8", "int4"]:
        runner.run_single(precision=prec)
    
    runner.print_summary(output_file=args.output)
