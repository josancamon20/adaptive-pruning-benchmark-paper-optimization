"""
APT (Adaptive Pruning and Tuning) Adapter Implementation
Based on the paper: "APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
import numpy as np


class APTAdapter(nn.Module):
    """
    APT Adapter that dynamically adjusts rank and dimensions during training
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 initial_rank: int = 8,
                 max_rank: int = 64,
                 reduction_factor: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.initial_rank = initial_rank
        self.max_rank = max_rank
        self.current_rank = initial_rank
        self.reduction_factor = reduction_factor
        
        # Initialize low-rank matrices with maximum possible rank
        self.lora_A = nn.Parameter(torch.randn(max_rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(out_features, max_rank))
        
        # Scaling factor
        self.scaling = 1.0 / initial_rank
        
        # Track which dimensions are active
        self.active_rank = initial_rank
        
    def forward(self, x: torch.Tensor, pruning_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with dynamic rank"""
        # Use only active dimensions
        active_A = self.lora_A[:self.active_rank, :]
        active_B = self.lora_B[:, :self.active_rank]
        
        if pruning_mask is not None:
            # Apply pruning mask to input dimensions
            x_masked = x * pruning_mask.unsqueeze(0)
            result = (x_masked @ active_A.T @ active_B.T) * self.scaling
        else:
            result = (x @ active_A.T @ active_B.T) * self.scaling
            
        return result
    
    def adjust_rank(self, new_rank: int):
        """Dynamically adjust the rank of the adapter"""
        new_rank = min(new_rank, self.max_rank)
        new_rank = max(new_rank, 1)
        
        if new_rank != self.active_rank:
            self.active_rank = new_rank
            self.scaling = 1.0 / new_rank
            
            # If increasing rank, initialize new parameters
            if new_rank > self.current_rank:
                with torch.no_grad():
                    # Initialize new rows in A
                    self.lora_A[self.current_rank:new_rank, :].normal_(0, 0.02)
                    # Keep B zeros for new dimensions
                    self.lora_B[:, self.current_rank:new_rank].zero_()
                    
            self.current_rank = max(self.current_rank, new_rank)


class OutlierAwareSalienceScoring:
    """
    Implements outlier-aware salience scoring for adaptive pruning
    """
    
    def __init__(self, tau: float = 4.0, ema_alpha: float = 0.85):
        self.tau = tau
        self.ema_alpha = ema_alpha
        self.salience_history: Dict[str, torch.Tensor] = {}
        
    def compute_salience(self, 
                        gradients: torch.Tensor, 
                        weights: torch.Tensor,
                        layer_name: str,
                        mu: float = 0.0) -> torch.Tensor:
        """
        Compute outlier-aware salience scores for pruning
        
        Args:
            gradients: Gradients w.r.t. the parameters
            weights: Parameter weights
            layer_name: Name of the layer for tracking history
            mu: Outlier coefficient (0 at start, 1 at end of pruning)
        """
        # Base salience score (magnitude of gradient * weight)
        base_salience = torch.abs(gradients * weights)
        
        # Compute outlier-aware adjustment
        if mu > 0:
            # Get statistics for outlier detection
            mean_salience = base_salience.mean()
            std_salience = base_salience.std()
            
            # Identify outliers
            outlier_threshold = mean_salience + self.tau * std_salience
            is_outlier = base_salience > outlier_threshold
            
            # Apply outlier weighting
            outlier_weight = torch.where(is_outlier, 
                                       torch.ones_like(base_salience),
                                       mu * torch.ones_like(base_salience))
            
            salience = base_salience * outlier_weight
        else:
            salience = base_salience
            
        # Apply exponential moving average
        if layer_name in self.salience_history:
            salience = (self.ema_alpha * self.salience_history[layer_name] + 
                       (1 - self.ema_alpha) * salience)
        
        self.salience_history[layer_name] = salience.detach().clone()
        
        return salience
    
    def get_pruning_mask(self, 
                        salience_scores: torch.Tensor, 
                        sparsity_ratio: float) -> torch.Tensor:
        """
        Generate binary pruning mask based on salience scores
        
        Args:
            salience_scores: Computed salience scores
            sparsity_ratio: Fraction of parameters to prune
        """
        if sparsity_ratio <= 0:
            return torch.ones_like(salience_scores)
        
        # Flatten scores for global thresholding
        flat_scores = salience_scores.flatten()
        
        # Find threshold for desired sparsity
        num_params = flat_scores.numel()
        num_prune = int(num_params * sparsity_ratio)
        
        if num_prune >= num_params:
            return torch.zeros_like(salience_scores)
        
        # Get threshold value
        threshold_value, _ = torch.kthvalue(flat_scores, num_prune + 1)
        
        # Create mask (1 = keep, 0 = prune)
        mask = (salience_scores >= threshold_value).float()
        
        return mask


class SelfDistillationLoss:
    """
    Implements self-distillation loss for APT training
    """
    
    def __init__(self, 
                 pred_weight: float = 1.0,
                 layer_weight: float = 0.9,
                 temperature: float = 4.0):
        self.pred_weight = pred_weight
        self.layer_weight = layer_weight
        self.temperature = temperature
        
    def compute_layer_mapping(self, 
                            teacher_layers: int, 
                            student_layers: int) -> List[int]:
        """
        Compute teacher-student layer mapping for distillation
        """
        if teacher_layers == student_layers:
            return list(range(student_layers))
        
        # Linear interpolation for mapping
        mapping = []
        for i in range(student_layers):
            teacher_idx = int(i * (teacher_layers - 1) / (student_layers - 1))
            mapping.append(teacher_idx)
            
        return mapping
    
    def compute_distillation_loss(self,
                                teacher_outputs: Dict[str, torch.Tensor],
                                student_outputs: Dict[str, torch.Tensor],
                                task_type: str = "classification") -> torch.Tensor:
        """
        Compute self-distillation loss
        
        Args:
            teacher_outputs: Outputs from unpruned model
            student_outputs: Outputs from pruned model  
            task_type: Type of task for loss weighting
        """
        total_loss = 0.0
        
        # Prediction distillation loss
        if "logits" in teacher_outputs and "logits" in student_outputs:
            teacher_logits = teacher_outputs["logits"] / self.temperature
            student_logits = student_outputs["logits"] / self.temperature
            
            pred_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction='batchmean'
            )
            
            total_loss += self.pred_weight * pred_loss
        
        # Layer-wise distillation loss
        if "hidden_states" in teacher_outputs and "hidden_states" in student_outputs:
            teacher_hidden = teacher_outputs["hidden_states"]
            student_hidden = student_outputs["hidden_states"]
            
            # Compute layer mapping
            mapping = self.compute_layer_mapping(
                len(teacher_hidden), len(student_hidden)
            )
            
            layer_loss = 0.0
            for s_idx, t_idx in enumerate(mapping):
                if s_idx < len(student_hidden) and t_idx < len(teacher_hidden):
                    s_hidden = student_hidden[s_idx]
                    t_hidden = teacher_hidden[t_idx]
                    
                    # MSE loss between hidden representations
                    layer_loss += F.mse_loss(s_hidden, t_hidden)
                    
            layer_loss /= len(mapping)
            total_loss += self.layer_weight * layer_loss
        
        # Adjust weights based on task type
        if task_type in ["squad", "cnn_dm"]:
            # For generation tasks, emphasize layer distillation
            total_loss = 0.1 * self.pred_weight * pred_loss + 0.9 * self.layer_weight * layer_loss
        else:
            # For classification, use standard weights
            total_loss = self.pred_weight * pred_loss + 0.9 * self.layer_weight * layer_loss
            
        return total_loss


def create_apt_model(base_model, 
                    target_sparsity: float = 0.6,
                    initial_rank: int = 8,
                    max_rank: int = 64):
    """
    Create APT-enhanced model with adaptive adapters
    
    Args:
        base_model: Base transformer model
        target_sparsity: Target pruning sparsity
        initial_rank: Initial rank for adapters
        max_rank: Maximum rank for adapters
    """
    
    # Add APT adapters to all linear layers
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear) and "classifier" not in name.lower():
            # Create APT adapter
            apt_adapter = APTAdapter(
                in_features=module.in_features,
                out_features=module.out_features,
                initial_rank=initial_rank,
                max_rank=max_rank
            )
            
            # Register as buffer to track it
            setattr(module, 'apt_adapter', apt_adapter)
            
            # Create pruning mask
            mask = torch.ones(module.weight.shape)
            setattr(module, 'pruning_mask', nn.Parameter(mask, requires_grad=False))
    
    # Freeze base model parameters
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Only adapter parameters are trainable
    trainable_params = []
    for name, module in base_model.named_modules():
        if hasattr(module, 'apt_adapter'):
            trainable_params.extend(module.apt_adapter.parameters())
    
    return base_model, trainable_params