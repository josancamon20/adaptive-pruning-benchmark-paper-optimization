"""
Baseline LoRA Implementation for Comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset
import json
import os
from tqdm import tqdm
import numpy as np
import time
from typing import Dict


class LoRALayer(nn.Module):
    """Standard LoRA layer implementation"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.scaling = alpha / rank
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def add_lora_to_model(model, rank: int = 16, alpha: float = 16.0):
    """Add LoRA adapters to all linear layers"""
    trainable_params = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "classifier" not in name.lower():
            # Create LoRA layer
            lora_layer = LoRALayer(
                in_features=module.in_features,
                out_features=module.out_features,
                rank=rank,
                alpha=alpha
            )
            
            # Add to module
            setattr(module, 'lora_layer', lora_layer)
            trainable_params.extend(lora_layer.parameters())
    
    # Freeze original parameters
    for param in model.parameters():
        param.requires_grad = False
    
    return model, trainable_params


class LoRATrainer:
    """Baseline LoRA trainer for comparison"""
    
    def __init__(self,
                 model_name: str = "roberta-base",
                 task_name: str = "sst2",
                 rank: int = 16,
                 alpha: float = 16.0,
                 learning_rate: float = 2e-4,
                 num_epochs: int = 10,
                 batch_size: int = 16,
                 output_dir: str = "./baseline_results",
                 seed: int = 42):
        
        self.model_name = model_name
        self.task_name = task_name
        self.rank = rank
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.seed = seed
        
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load components
        self._load_model_and_tokenizer()
        self._load_dataset()
        self._setup_lora_model()
        
        self.metrics_log = {}
        
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer"""
        print(f"Loading baseline model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Set number of labels
        if self.task_name == "sst2":
            num_labels = 2
        elif self.task_name == "mnli":
            num_labels = 3
        else:
            num_labels = 2
            
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        ).to(self.device)
        
    def _load_dataset(self):
        """Load and tokenize dataset"""
        print(f"Loading dataset: {self.task_name}")
        
        if self.task_name == "sst2":
            dataset = load_dataset("glue", "sst2")
            self.train_dataset = dataset["train"]
            self.eval_dataset = dataset["validation"]
            text_column = "sentence"
        elif self.task_name == "mnli":
            dataset = load_dataset("glue", "mnli")
            self.train_dataset = dataset["train"]
            self.eval_dataset = dataset["validation_matched"]
            text_column = ["premise", "hypothesis"]
        
        def tokenize_function(examples):
            if self.task_name == "sst2":
                return self.tokenizer(
                    examples["sentence"],
                    truncation=True,
                    padding="max_length",
                    max_length=512
                )
            else:  # mnli
                return self.tokenizer(
                    examples["premise"],
                    examples["hypothesis"],
                    truncation=True,
                    padding="max_length",
                    max_length=512
                )
        
        self.train_dataset = self.train_dataset.map(tokenize_function, batched=True)
        self.eval_dataset = self.eval_dataset.map(tokenize_function, batched=True)
        
        self.train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        self.eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
    def _setup_lora_model(self):
        """Setup LoRA model"""
        print("Setting up LoRA model...")
        
        self.model, self.trainable_params = add_lora_to_model(
            self.model, rank=self.rank, alpha=self.alpha
        )
        
        self.optimizer = optim.AdamW(self.trainable_params, lr=self.learning_rate)
        
        total_steps = len(self.train_dataset) // self.batch_size * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        print(f"Trainable parameters: {sum(p.numel() for p in self.trainable_params):,}")
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        self.model.train()
        
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Apply LoRA layers in forward pass
        def lora_forward(module, input_tensor):
            if hasattr(module, 'lora_layer'):
                original_output = module._old_forward(input_tensor)
                lora_output = module.lora_layer(input_tensor)
                return original_output + lora_output
            return module._old_forward(input_tensor)
        
        # Monkey patch forward methods
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_layer') and not hasattr(module, '_old_forward'):
                module._old_forward = module.forward
                module.forward = lambda x, m=module: lora_forward(m, x)
        
        outputs = self.model(**batch)
        loss = outputs.loss
        
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.batch_size, shuffle=False)
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == batch["label"]).sum().item()
                total += batch["label"].size(0)
                
        return {
            "eval_loss": total_loss / len(eval_dataloader),
            "eval_accuracy": correct / total
        }
        
    def measure_efficiency(self) -> Dict[str, float]:
        """Measure efficiency metrics"""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.trainable_params)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
        else:
            memory_gb = 0
            
        # Inference speed
        dummy_input = {
            "input_ids": torch.randint(0, 1000, (1, 512)).to(self.device),
            "attention_mask": torch.ones(1, 512).to(self.device)
        }
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(**dummy_input)
                
        # Time inference
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        for _ in range(100):
            with torch.no_grad():
                _ = self.model(**dummy_input)
                
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / 100
        throughput = 1.0 / avg_inference_time
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "memory_gb": memory_gb,
            "inference_time_ms": avg_inference_time * 1000,
            "throughput_samples_per_sec": throughput
        }
        
    def train(self):
        """Main training loop"""
        print("Starting LoRA baseline training...")
        
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            epoch_losses = []
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                
            # Evaluate
            eval_metrics = self.evaluate()
            efficiency_metrics = self.measure_efficiency()
            
            # Log results
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": np.mean(epoch_losses),
                **eval_metrics,
                **efficiency_metrics
            }
            
            for key, value in epoch_metrics.items():
                if key not in self.metrics_log:
                    self.metrics_log[key] = []
                self.metrics_log[key].append(value)
                
            print(f"  Train Loss: {epoch_metrics['train_loss']:.4f}")
            print(f"  Eval Accuracy: {epoch_metrics['eval_accuracy']:.4f}")
            print(f"  Memory: {epoch_metrics['memory_gb']:.2f} GB")
            
    def save_results(self):
        """Save results"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        results_file = os.path.join(self.output_dir, f"lora_baseline_{self.task_name}.json")
        with open(results_file, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)
            
        print(f"Baseline results saved to {results_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--task_name", type=str, default="sst2")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./baseline_results")
    
    args = parser.parse_args()
    
    trainer = LoRATrainer(**vars(args))
    trainer.train()
    trainer.save_results()


if __name__ == "__main__":
    main()