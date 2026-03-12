"""
fedgrpo_energy.py
=================
Energy-aware Federated GRPO (Group Relative Policy Optimization).

This script simulates a federated learning environment where clients (mobile/edge devices)
perform RL training (GRPO) on a small model (Qwen3-0.6B) and the server aggregates
the results while managing an energy budget.

Based on:
  GRPO (DeepSeek-V3 / OpenR1)
  Energy-aware Federated Learning (Simmhan Lab, IISc)
"""

import time
import torch
import numpy as np
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from power_monitor import PowerMonitor

@dataclass
class FGRPOConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    n_clients: int = 2 # Reduced for faster demo
    rounds: int = 3
    energy_budget_joules: float = 1000.0
    n_rollouts_per_prompt: int = 4
    max_rollout_tokens: int = 32
    rollout_temperature: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class EnergyPredictor:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.p_base_watts = 0.0
        self.p_rollout_watts = 0.0
        self.tokens_per_second = 0.0
        self.time_per_grpo_step = 0.0
        self.monitor = PowerMonitor(sample_interval_ms=100)

    def calibrate(self, n_warmup=3):
        print("Calibrating energy predictor...")
        # 1. Base power (idle with model loaded)
        self.monitor.start()
        time.sleep(2)
        stats = self.monitor.stop()
        self.p_base_watts = stats.mean_w
        
        # 2. Rollout power and speed
        prompt = "Explain quantum physics in one sentence."
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        self.monitor.start()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=32, min_new_tokens=32, do_sample=True)
        t1 = time.perf_counter()
        stats = self.monitor.stop()
        
        self.p_rollout_watts = stats.mean_w
        self.tokens_per_second = 32 / (t1 - t0)
        
        # 3. GRPO step time (mock backward)
        t0 = time.perf_counter()
        loss = self.model(**inputs, labels=inputs["input_ids"]).loss
        loss.backward()
        self.time_per_grpo_step = time.perf_counter() - t0
        
        print(f"Calibration complete: Base={self.p_base_watts:.2f}W, Rollout={self.p_rollout_watts:.2f}W, Speed={self.tokens_per_second:.2f} tps")

    def predict_energy(self, n_prompts, expected_rollout_length, n_grpo_steps):
        # Energy = (Power * Time)
        rollout_time = (n_prompts * expected_rollout_length) / self.tokens_per_second
        backward_time = n_grpo_steps * self.time_per_grpo_step
        
        e_rollout = self.p_rollout_watts * rollout_time
        e_backward = self.p_rollout_watts * 1.2 * backward_time # Assume 20% more power for backward
        return e_rollout + e_backward

    def measure_actual_energy(self, fn, *args, **kwargs):
        self.monitor.start()
        result = fn(*args, **kwargs)
        stats = self.monitor.stop()
        return result, stats.energy_joules

class GRPOClient:
    def __init__(self, client_id, config, dataset, shared_model, tokenizer):
        self.client_id = client_id
        self.config = config
        self.dataset = dataset # List of {"question": str, "answer": str}
        self.model = shared_model
        self.tokenizer = tokenizer
        self.energy_used = 0.0
        # Client-specific LoRA weights
        self.local_weights = {k: v.cpu().clone() for k, v in shared_model.state_dict().items() if "lora" in k}

    def _generate_rollouts(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_rollout_tokens,
                num_return_sequences=self.config.n_rollouts_per_prompt,
                do_sample=True,
                temperature=self.config.rollout_temperature,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract completions (removing prompt tokens)
        prompt_len = inputs["input_ids"].shape[1]
        completions = [self.tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in outputs]
        return completions

    def _compute_reward(self, response, correct_answer):
        # Simple binary reward: 1.0 if correct answer is in response, else 0.0
        return 1.0 if str(correct_answer).lower() in response.lower() else 0.0

    def _grpo_loss(self, prompt, completions, advantages):
        # Highly simplified GRPO surrogate loss implementation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        # Loss scaled by mean advantage
        loss = outputs.loss * torch.tensor(advantages).mean().to(self.config.device)
        return loss

    def train_round(self, n_steps=1):
        # Load local weights before training
        self.model.load_state_dict(self.local_weights, strict=False)
        
        predictor = EnergyPredictor(self.model, self.tokenizer, self.config.device)
        predictor.calibrate()
        
        def _step():
            total_r = 0
            for item in self.dataset[:n_steps]:
                prompt = item["question"]
                answer = item["answer"]
                
                completions = self._generate_rollouts(prompt)
                rewards = [self._compute_reward(c, answer) for c in completions]
                
                # GRPO advantage = (reward - mean(rewards_in_group))
                mean_r = sum(rewards) / len(rewards)
                advantages = [r - mean_r for r in rewards]
                
                loss = self._grpo_loss(prompt, completions, advantages)
                loss.backward()
                total_r += mean_r
            return total_r / n_steps

        avg_reward, energy = predictor.measure_actual_energy(_step)
        self.energy_used += energy
        
        # Save updated weights
        self.local_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items() if "lora" in k}
        
        return {
            "client_id": self.client_id,
            "avg_reward": avg_reward,
            "energy_joules": energy,
            "weights": self.local_weights
        }

class EnergyAwareClientSelector:
    def __init__(self, n_clients, per_round_budget):
        self.n_clients = n_clients
        self.per_round_budget = per_round_budget
        self._reward_history = {i: [0.1] for i in range(n_clients)} # Seed with small value
        self._energy_history = {i: [50.0] for i in range(n_clients)}

    def estimate_shapley(self, client_id):
        # Simplified: Use historical mean reward as importance score
        return sum(self._reward_history[client_id]) / len(self._reward_history[client_id])

    def predict_client_energy(self, client_id):
        return sum(self._energy_history[client_id]) / len(self._energy_history[client_id])

    def select_clients(self, available_clients):
        # Knapsack-style greedy selection: Maximize reward/importance within energy budget
        scores = []
        for c in available_clients:
            importance = self.estimate_shapley(c.client_id)
            energy = self.predict_client_energy(c.client_id)
            scores.append((importance / energy, c))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        
        selected = []
        remaining_budget = self.per_round_budget
        for ratio, client in scores:
            pred_e = self.predict_client_energy(client.client_id)
            if pred_e <= remaining_budget:
                selected.append(client)
                remaining_budget -= pred_e
        
        return selected

    def update_history(self, client_id, reward, energy):
        self._reward_history[client_id].append(reward)
        self._energy_history[client_id].append(energy)

class FedGRPOServer:
    def __init__(self, config):
        self.config = config
        self.selector = EnergyAwareClientSelector(config.n_clients, config.energy_budget_joules / config.rounds)
        self.global_weights = None
        self.total_energy_used = 0.0

    def aggregate(self, client_results):
        if not client_results:
            return
        
        # Simple Federated Averaging of LoRA weights
        new_weights = {}
        n = len(client_results)
        for res in client_results:
            self.selector.update_history(res["client_id"], res["avg_reward"], res["energy_joules"])
            self.total_energy_used += res["energy_joules"]
            
            for k, v in res["weights"].items():
                if k not in new_weights:
                    new_weights[k] = v / n
                else:
                    new_weights[k] += v / n
        
        self.global_weights = new_weights

    def run_round(self, clients, round_idx):
        print(f"\n--- Round {round_idx+1}/{self.config.rounds} ---")
        # 1. Select clients based on energy budget
        selected_clients = self.selector.select_clients(clients)
        print(f"Selected {len(selected_clients)}/{len(clients)} clients")
        
        # 2. Distribute global weights (if any)
        if self.global_weights:
            for c in selected_clients:
                c.model.load_state_dict(self.global_weights, strict=False)
        
        # 3. Local training
        results = []
        for c in selected_clients:
            print(f"  Training Client {c.client_id}...")
            res = c.train_round(n_steps=1)
            results.append(res)
            print(f"    Reward: {res['avg_reward']:.2f}, Energy: {res['energy_joules']:.2f}J")
            
        # 4. Aggregate
        self.aggregate(results)
        print(f"Total energy used so far: {self.total_energy_used:.2f}J / {self.config.energy_budget_joules}J")

def run_benchmark():
    config = FGRPOConfig()
    print(f"Starting Energy-aware Federated GRPO with {config.n_clients} clients...")
    
    # 1. Load shared base model and wrap with LoRA
    print(f"Loading shared model {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.float16, device_map=config.device
    )
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    shared_model = get_peft_model(base_model, lora_config)
    
    # 2. Setup dataset
    dataset = [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Capital of UK?", "answer": "London"},
        {"question": "Color of sky?", "answer": "blue"},
    ]
    
    # 3. Create clients with shared model
    clients = []
    for i in range(config.n_clients):
        c = GRPOClient(i, config, dataset, shared_model, tokenizer)
        clients.append(c)
        
    server = FedGRPOServer(config)
    
    for r in range(config.rounds):
        server.run_round(clients, r)
        if server.total_energy_used >= config.energy_budget_joules:
            print("Energy budget exhausted!")
            break
            
    print("\nBenchmark Finished.")
    print(f"Final Total Energy: {server.total_energy_used:.2f} J")

if __name__ == "__main__":
    run_benchmark()
