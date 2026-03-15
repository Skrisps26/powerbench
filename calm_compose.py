"""
calm_compose.py
===============
Implementation of CALM (Composing Augmentation-tuned Language Models) for Qwen3-0.6B.

Based on:
  CALM (Talukdar lab, 2024)
  Idea: Take a base "anchor" model and a domain-specific "augment" model.
  Use cross-attention bridges to fuse representations from the augment model
  into the anchor model without full fine-tuning.

This file implements the CrossAttentionBridge and the CALMComposer model.
"""

import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class CrossAttentionBridge(nn.Module):
    """
    Fuses hidden states from an augment model into the anchor model's stream.
    """
    def __init__(self, anchor_dim, augment_dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = anchor_dim // n_heads
        
        self.W_q = nn.Linear(anchor_dim, anchor_dim)
        self.W_k = nn.Linear(augment_dim, anchor_dim)
        self.W_v = nn.Linear(augment_dim, anchor_dim)
        self.W_o = nn.Linear(anchor_dim, anchor_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(anchor_dim)

    def _split_heads(self, x):
        batch, seq_len, dim = x.shape
        return x.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2).contiguous()

    def _merge_heads(self, x):
        batch, n_heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, n_heads * head_dim)

    def forward(self, anchor_hidden, augment_hidden, attention_mask=None):
        # print(f"anchor_hidden shape: {anchor_hidden.shape}")
        # print(f"augment_hidden shape: {augment_hidden.shape}")
        Q = self.W_q(anchor_hidden)
        K = self.W_k(augment_hidden)
        V = self.W_v(augment_hidden)
        
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # head_dim is used for scaling
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        
        if attention_mask is not None:
            # scores shape: (batch, n_heads, seq_len, seq_len)
            # mask shape should be (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
            mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=scores.dtype)
            mask = (1.0 - mask) * torch.finfo(scores.dtype).min
            scores = scores + mask
            
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = self._merge_heads(torch.matmul(weights, V))
        return self.layer_norm(anchor_hidden + self.W_o(attended))

class CALMComposer(nn.Module):
    """
    Wraps two models and bridges them.
    """
    def __init__(self, anchor_model, augment_model, bridge_layers=None):
        super().__init__()
        self.anchor = anchor_model
        self.augment = augment_model
        
        # bridge_layers is a list of layer indices where we bridge
        if bridge_layers is None:
            bridge_layers = [4, 8, 12] # Default for Qwen3-0.6B (24 layers)
        
        self.bridge_layers = nn.ModuleDict({
            str(i): CrossAttentionBridge(
                anchor_model.config.hidden_size,
                augment_model.config.hidden_size,
                n_heads=anchor_model.config.num_attention_heads
            ) for i in bridge_layers
        })

    def _get_layer_hidden_states(self, model, input_ids, attention_mask=None):
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return list(outputs.hidden_states[1:])

    def forward(self, input_ids, attention_mask=None, labels=None):
        # 1. Get hidden states from both models
        # We need output_hidden_states=True
        with torch.no_grad():
            aug_outputs = self.augment(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            aug_hiddens = aug_outputs.hidden_states[1:] # 0 is embeddings
            
            # For the anchor, we need to be careful. CALM usually modifies the anchor forward.
            # But if we just want to benchmark the *bridge* logic:
            anchor_outputs = self.anchor(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            anchor_hiddens = list(anchor_outputs.hidden_states[1:])
        
        # 2. Apply bridging at specified layers
        # In a real CALM, the bridged state at layer i affects layer i+1.
        # This simplified version just demonstrates the bridge calculation.
        seq_len = input_ids.size(1)
        
        for layer_idx_str, bridge in self.bridge_layers.items():
            idx = int(layer_idx_str)
            if idx < len(anchor_hiddens):
                # Apply bridge to the hidden state produced by layer 'idx'
                # Move augment hidden to same device as anchor hidden
                aug_h = aug_hiddens[idx].to(anchor_hiddens[idx].device)
                anchor_hiddens[idx] = bridge(anchor_hiddens[idx], aug_h, attention_mask)
        
        # Use the last hidden state for logits
        hidden = anchor_hiddens[-1]
        logits = self.anchor.lm_head(hidden)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.anchor.config.vocab_size), shift_labels.view(-1))
            
        return {"loss": loss, "logits": logits}

    def evaluate_perplexity(self, eval_dataloader, device="cuda"):
        self.eval()
        total_loss = 0.0
        total_steps = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = self.forward(**inputs, labels=inputs["input_ids"])
                total_loss += outputs["loss"].item()
                total_steps += 1
        
        mean_loss = total_loss / total_steps if total_steps > 0 else 0.0
        self.train()
        return float(torch.exp(torch.tensor(mean_loss)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    model_path = args.model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading anchor in 4-bit...")
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    
    anchor = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map=device)
    # Mocking augment model as anchor itself to demonstrate bridge logic without double memory
    augment = anchor 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    composer = CALMComposer(anchor, augment).to(device)
    # Match the model's compute dtype (usually bfloat16 for Qwen3 4-bit)
    if hasattr(anchor, "dtype"):
        composer.to(anchor.dtype)
    print("CALM Composer initialized with bridges at layers:", list(composer.bridge_layers.keys()))

    # Simple evaluation on a few sentences
    eval_texts = [
        "The theory of relativity was proposed by Albert Einstein.",
        "The capital of France is Paris.",
        "Machine learning is a subset of artificial intelligence."
    ]
    
    # Mock a dataloader-like structure
    class MockDataloader:
        def __init__(self, texts, tokenizer):
            self.inputs = [tokenizer(t, return_tensors="pt") for t in texts]
        def __iter__(self):
            return iter(self.inputs)

    print("Running CALM benchmark evaluation...")
    dataloader = MockDataloader(eval_texts, tokenizer)
    try:
        perplexity = composer.evaluate_perplexity(dataloader, device=device)
        print(f"CALM Composite Perplexity: {perplexity:.4f}")
    except Exception as e:
        print(f"Benchmark failed: {e}")
