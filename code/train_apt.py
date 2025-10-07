"""
APT Training Script
Reproduces the main experiments from the APT paper
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup, Trainer, TrainingArguments
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict

from apt_adapter import (
    APTAdapter, OutlierAwareSalienceScoring, SelfDistillationLoss,
    create_apt_model
)


class APTTrainer:
    """Main trainer class for APT experiments"""
    
    def __init__(self, 
                 model_name: str = "roberta-base",
                 task_name: str = "sst2",
                 target_sparsity: float = 0.6,
                 initial_rank: int = 8,
                 max_rank: int = 64,
                 learning_rate: float = 2e-4,
                 num_epochs: int = 10,
                 batch_size: int = 16,
                 output_dir: str = "./results",
                 seed: int = 42):
        
        self.model_name = model_name
        self.task_name = task_name
        self.target_sparsity = target_sparsity
        self.initial_rank = initial_rank
        self.max_rank = max_rank
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize components
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.salience_scorer = OutlierAwareSalienceScoring()
        self.distillation_loss = SelfDistillationLoss()
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Load dataset
        self._load_dataset()
        
        # Setup model
        self._setup_apt_model()
        
        # Metrics tracking
        self.metrics_log = defaultdict(list)
        
    def _load_model_and_tokenizer(self):
        """Load the base model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Determine number of labels based on task
        if self.task_name in ["sst2"]:
            num_labels = 2
        elif self.task_name in ["mnli"]:
            num_labels = 3
        else:
            num_labels = 2  # Default
            
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=False
        )
        
        # Move to device
        self.base_model = self.base_model.to(self.device)
        
    def _load_dataset(self):
        """Load and preprocess dataset"""
        print(f"Loading dataset: {self.task_name}")
        
        if self.task_name == "sst2":
            dataset = load_dataset("glue", "sst2")
            self.train_dataset = dataset["train"]
            self.eval_dataset = dataset["validation"]
            self.text_column = "sentence"
            self.label_column = "label"
            
        elif self.task_name == "mnli":
            dataset = load_dataset("glue", "mnli")
            self.train_dataset = dataset["train"]
            self.eval_dataset = dataset["validation_matched"]
            self.text_column = ["premise", "hypothesis"]
            self.label_column = "label"
        else:
            raise ValueError(f"Task {self.task_name} not supported")
            
        # Tokenize dataset
        self._tokenize_dataset()
        
    def _tokenize_dataset(self):
        """Tokenize the datasets"""
        def tokenize_function(examples):
            if self.task_name == "sst2":
                return self.tokenizer(
                    examples[self.text_column],
                    truncation=True,
                    padding="max_length",
                    max_length=512
                )
            elif self.task_name == "mnli":
                return self.tokenizer(
                    examples["premise"],
                    examples["hypothesis"],
                    truncation=True,
                    padding="max_length",
                    max_length=512
                )
                
        self.train_dataset = self.train_dataset.map(tokenize_function, batched=True)
        self.eval_dataset = self.eval_dataset.map(tokenize_function, batched=True)
        
        # Set format for PyTorch
        self.train_dataset.set_format(
            type="torch", 
            columns=["input_ids", "attention_mask", "label"]
        )
        self.eval_dataset.set_format(
            type="torch", 
            columns=["input_ids", "attention_mask", "label"]
        )
        
    def _setup_apt_model(self):
        """Setup APT model with adapters"""
        print("Setting up APT model...")
        
        # Create APT-enhanced model
        self.model, self.trainable_params = create_apt_model(
            self.base_model,
            target_sparsity=self.target_sparsity,
            initial_rank=self.initial_rank,
            max_rank=self.max_rank
        )
        
        # Setup optimizer only for trainable parameters
        self.optimizer = optim.AdamW(self.trainable_params, lr=self.learning_rate)
        
        # Calculate total training steps
        total_steps = len(self.train_dataset) // self.batch_size * self.num_epochs
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        print(f"Trainable parameters: {sum(p.numel() for p in self.trainable_params):,}")
        
    def _compute_mu(self, current_step: int, total_steps: int, 
                   pruning_start: float = 0.1, pruning_end: float = 0.8) -> float:
        """Compute mu parameter for outlier-aware salience"""
        start_step = int(pruning_start * total_steps)
        end_step = int(pruning_end * total_steps)
        
        if current_step < start_step:
            return 0.0
        elif current_step > end_step:
            return 1.0
        else:
            return (current_step - start_step) / (end_step - start_step)
            
    def _compute_sparsity_schedule(self, current_step: int, total_steps: int,
                                 pruning_start: float = 0.1, pruning_end: float = 0.8) -> float:
        """Compute sparsity ratio for current step"""
        start_step = int(pruning_start * total_steps)
        end_step = int(pruning_end * total_steps)
        
        if current_step < start_step:
            return 0.0
        elif current_step > end_step:
            return self.target_sparsity
        else:
            progress = (current_step - start_step) / (end_step - start_step)
            return progress * self.target_sparsity
            
    def _update_pruning_masks(self, current_step: int, total_steps: int):
        """Update pruning masks based on salience scores"""
        mu = self._compute_mu(current_step, total_steps)
        current_sparsity = self._compute_sparsity_schedule(current_step, total_steps)
        
        if current_sparsity == 0:
            return
            
        # Collect gradients and compute salience
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'pruning_mask'):
                if module.weight.grad is not None:
                    # Compute salience scores
                    salience = self.salience_scorer.compute_salience(
                        gradients=module.weight.grad,
                        weights=module.weight.data,
                        layer_name=name,
                        mu=mu
                    )
                    
                    # Update pruning mask
                    new_mask = self.salience_scorer.get_pruning_mask(
                        salience, current_sparsity
                    )
                    
                    module.pruning_mask.data = new_mask
                    
    def _adaptive_rank_adjustment(self, current_step: int, total_steps: int):
        """Adaptively adjust adapter ranks based on layer importance"""
        # Simple heuristic: gradually increase rank in early training
        progress = current_step / total_steps
        
        if progress < 0.3:
            # Early training: increase rank for important layers
            target_rank = min(self.max_rank, int(self.initial_rank * (1 + progress)))
        else:
            # Later training: stabilize or reduce rank
            target_rank = max(self.initial_rank, int(self.max_rank * (1 - 0.5 * (progress - 0.3))))
            
        # Update adapter ranks
        for name, module in self.model.named_modules():
            if hasattr(module, 'apt_adapter'):
                module.apt_adapter.adjust_rank(target_rank)
                
    def train_step(self, batch: Dict[str, torch.Tensor], step: int, total_steps: int) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with pruned model (student)
        student_outputs = self.model(**batch)
        student_loss = student_outputs.loss
        
        # Forward pass with unpruned model (teacher) for distillation
        with torch.no_grad():
            # Temporarily remove pruning masks
            original_masks = {}
            for name, module in self.model.named_modules():
                if hasattr(module, 'pruning_mask'):
                    original_masks[name] = module.pruning_mask.clone()
                    module.pruning_mask.data.fill_(1.0)  # No pruning
                    
            teacher_outputs = self.model(**batch)
            
            # Restore pruning masks
            for name, module in self.model.named_modules():
                if name in original_masks:
                    module.pruning_mask.data = original_masks[name]
        
        # Compute distillation loss
        distillation_loss = self.distillation_loss.compute_distillation_loss(
            teacher_outputs=teacher_outputs,
            student_outputs=student_outputs,
            task_type="classification" if self.task_name in ["sst2", "mnli"] else "other"
        )
        
        # Total loss
        total_loss = student_loss + distillation_loss
        
        # Backward pass
        total_loss.backward()
        
        # Update pruning masks
        self._update_pruning_masks(step, total_steps)
        
        # Adaptive rank adjustment
        self._adaptive_rank_adjustment(step, total_steps)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return {
            "total_loss": total_loss.item(),
            "task_loss": student_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "learning_rate": self.scheduler.get_last_lr()[0]
        }
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        eval_dataloader = DataLoader(
            self.eval_dataset, 
            batch_size=self.batch_size,
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                total_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct_predictions += (predictions == batch["label"]).sum().item()
                total_predictions += batch["label"].size(0)
                
        avg_loss = total_loss / len(eval_dataloader)
        accuracy = correct_predictions / total_predictions
        
        return {
            "eval_loss": avg_loss,
            "eval_accuracy": accuracy
        }
        
    def measure_efficiency(self) -> Dict[str, float]:
        """Measure training and inference efficiency"""
        # Count active parameters
        active_params = 0
        total_params = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                total_params += module.weight.numel()
                if hasattr(module, 'pruning_mask'):
                    active_params += module.pruning_mask.sum().item()
                else:
                    active_params += module.weight.numel()
                    
        # Add adapter parameters
        adapter_params = sum(p.numel() for p in self.trainable_params)
        
        # Measure memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9  # GB
        else:
            memory_allocated = 0
            
        # Measure inference speed
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
            "active_params": active_params,
            "adapter_params": adapter_params,
            "sparsity_ratio": 1.0 - (active_params / total_params),
            "memory_gb": memory_allocated,
            "inference_time_ms": avg_inference_time * 1000,
            "throughput_samples_per_sec": throughput
        }
        
    def train(self):
        """Main training loop"""
        print("Starting APT training...")
        
        # Create data loader
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        total_steps = len(train_dataloader) * self.num_epochs
        step = 0
        
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            epoch_losses = []
            progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for batch in progress_bar:
                # Training step
                metrics = self.train_step(batch, step, total_steps)
                epoch_losses.append(metrics["total_loss"])
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{metrics['total_loss']:.4f}",
                    "lr": f"{metrics['learning_rate']:.2e}"
                })
                
                step += 1
                
            # Evaluate at end of epoch
            eval_metrics = self.evaluate()
            
            # Measure efficiency
            efficiency_metrics = self.measure_efficiency()
            
            # Log metrics
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": np.mean(epoch_losses),
                **eval_metrics,
                **efficiency_metrics
            }
            
            for key, value in epoch_metrics.items():
                self.metrics_log[key].append(value)
                
            print(f"Epoch {epoch + 1} Results:")
            print(f"  Train Loss: {epoch_metrics['train_loss']:.4f}")
            print(f"  Eval Loss: {epoch_metrics['eval_loss']:.4f}")
            print(f"  Eval Accuracy: {epoch_metrics['eval_accuracy']:.4f}")
            print(f"  Sparsity: {epoch_metrics['sparsity_ratio']:.2%}")
            print(f"  Memory: {epoch_metrics['memory_gb']:.2f} GB")
            print(f"  Throughput: {epoch_metrics['throughput_samples_per_sec']:.2f} samples/sec")
            
    def save_results(self):
        """Save training results"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save metrics
        metrics_file = os.path.join(self.output_dir, f"metrics_{self.task_name}_{self.model_name.replace('/', '_')}.json")
        with open(metrics_file, 'w') as f:
            json.dump(dict(self.metrics_log), f, indent=2)
            
        # Save final model state
        model_file = os.path.join(self.output_dir, f"model_{self.task_name}_{self.model_name.replace('/', '_')}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'model_name': self.model_name,
                'task_name': self.task_name,
                'target_sparsity': self.target_sparsity,
                'initial_rank': self.initial_rank,
                'max_rank': self.max_rank,
            }
        }, model_file)
        
        print(f"Results saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train APT model")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--task_name", type=str, default="sst2", choices=["sst2", "mnli"])
    parser.add_argument("--target_sparsity", type=float, default=0.6)
    parser.add_argument("--initial_rank", type=int, default=8)
    parser.add_argument("--max_rank", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = APTTrainer(**vars(args))
    
    # Train model
    trainer.train()
    
    # Save results
    trainer.save_results()


if __name__ == "__main__":
    main()