"""
benchmark_inference.py
======================
Benchmarks LLM inference latency and perplexity across different quantization levels.
"""

import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
            
        sorted_lat = sorted(latencies)
        mean_lat = sum(latencies) / len(latencies)
        return {
            "mean_latency_ms": mean_lat,
            "p50_latency_ms": sorted_lat[len(sorted_lat)//2],
            "p95_latency_ms": sorted_lat[int(0.95*len(sorted_lat))],
            "tokens_per_second": batch_size * output_length / (mean_lat / 1000)
        }

class PerplexityMeasurer:
    EVAL_TEXTS = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world of technology.",
        "Quantum computing relies on the principles of superposition and entanglement."
    ]

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def measure(self):
        total_loss = 0.0
        for text in self.EVAL_TEXTS:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
        
        mean_loss = total_loss / len(self.EVAL_TEXTS)
        return float(torch.exp(torch.tensor(mean_loss)))

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for prec in ["fp16", "int8", "int4"]:
        print(f"\nBenchmarking {prec}...")
        try:
            model, tokenizer = ModelLoader.load(model_name, precision=prec, device=device)
            size = ModelLoader.model_size_mb(model)
            print(f"Model size: {size:.2f} MB")
            
            latency_measurer = LatencyMeasurer(model, tokenizer, device)
            lat_stats = latency_measurer.measure(batch_size=1, input_length=32, output_length=32)
            print(f"Latency stats: {lat_stats}")
            
            ppl_measurer = PerplexityMeasurer(model, tokenizer, device)
            ppl = ppl_measurer.measure()
            print(f"Perplexity: {ppl:.2f}")
            
            ModelLoader.unload(model)
        except Exception as e:
            print(f"Failed to benchmark {prec}: {e}")
