# APT: Adaptive Pruning and Tuning - Paper Reproduction

This repository contains a reproduction of the main experiments from the paper:

**"APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference"**

## Overview

This reproduction implements the core APT (Adaptive Pruning and Tuning) methodology that combines parameter-efficient fine-tuning with structured pruning to achieve both training and inference efficiency improvements.

### Key Components Implemented

1. **APT Adapter** (`apt_adapter.py`):
   - Dynamic rank adjustment during training
   - Low-rank decomposition with adaptive dimensions
   - Integration with pruning masks

2. **Outlier-Aware Salience Scoring** (`apt_adapter.py`):
   - Exponential moving average (α=0.85) for salience tracking
   - Outlier detection with threshold τ=4
   - Binary mask generation for structured pruning

3. **Self-Distillation Loss** (`apt_adapter.py`):
   - Teacher-student layer mapping
   - Prediction and layer-wise distillation
   - Task-specific loss weighting

4. **Training Framework** (`train_apt.py`):
   - Dynamic pruning schedule during training
   - Adaptive rank adjustment based on training progress
   - Comprehensive efficiency measurements

5. **Baseline Implementation** (`baseline_lora.py`):
   - Standard LoRA implementation for comparison
   - Same training setup and evaluation metrics

## Experiments Reproduced

Based on the paper's main contributions, this reproduction focuses on:

### Core Experiments
1. **RoBERTa-base on SST2** with 60% sparsity
2. **RoBERTa-base on MNLI** with 60% sparsity  
3. **LoRA baselines** on both tasks for comparison
4. **Multiple sparsity levels** (40%, 60%, 80%) to show scalability

### Metrics Measured
- **Task Performance**: Accuracy on dev sets
- **Training Efficiency**: Memory usage, training time
- **Inference Efficiency**: Throughput (samples/sec), inference time
- **Model Compression**: Sparsity ratios, parameter counts

## Files Structure

```
code/
├── reproduce.sh              # Main reproduction script
├── requirements.txt           # Python dependencies
├── apt_adapter.py            # Core APT implementation
├── train_apt.py              # APT training script
├── baseline_lora.py          # LoRA baseline implementation
├── README.md                 # This file
├── results/                  # APT experiment results
├── baseline_results/         # LoRA baseline results
└── comparison_results/       # Comparative analysis
```

## Usage

### Quick Start
```bash
# Run all experiments
bash reproduce.sh
```

### Individual Experiments
```bash
# APT on SST2 with 60% sparsity
python3 train_apt.py --model_name roberta-base --task_name sst2 --target_sparsity 0.6

# LoRA baseline on SST2
python3 baseline_lora.py --model_name roberta-base --task_name sst2

# APT with different sparsity
python3 train_apt.py --target_sparsity 0.8 --task_name sst2
```

## Implementation Details

### APT Algorithm
1. **Initialization**: Start with low-rank adapters (rank=8)
2. **Early Training**: Gradually add tuning parameters and begin pruning
3. **Mid Training**: Compute outlier-aware salience scores with increasing μ
4. **Late Training**: Stabilize pruning masks and reduce adapter ranks
5. **Throughout**: Apply self-distillation between pruned and unpruned models

### Key Parameters
- **Target Sparsity**: 60% (main experiments)
- **Initial Rank**: 8, Max Rank: 64
- **EMA Alpha**: 0.85 for salience scoring
- **Tau**: 4.0 for outlier detection
- **Learning Rate**: 2e-4
- **Distillation Weights**: Task-dependent (see addendum)

### Hyperparameters Following Paper/Addendum
- **Outlier-aware salience**: Exponential moving average with α=0.85
- **μ schedule**: Linear from 0 to 1 during pruning phase
- **Teacher-student mapping**: Recomputed every step
- **Distillation loss weights**: 
  - Classification: L_pred + 0.9 * L_layer
  - Generation: 0.1 * L_pred + 0.9 * L_layer

## Expected Results

Based on the paper's reported results, this reproduction should achieve:

### Performance Targets
- **RoBERTa SST2**: ~98% of baseline accuracy with 60% sparsity
- **RoBERTa MNLI**: ~98% of baseline accuracy with 60% sparsity
- **Memory Reduction**: ~70% training memory footprint reduction
- **Speed Improvement**: 2-8x training speedup, 2.4x inference speedup

### Output Files
After running `reproduce.sh`, you will find:

1. **Individual Results**: JSON files with detailed metrics per experiment
2. **Summary Table**: CSV with comparative results across all experiments  
3. **Detailed Analysis**: Complete results in JSON format
4. **Console Output**: Real-time progress and final comparison summary

## Limitations and Scope

### What's Included
- Core APT methodology with all key components
- RoBERTa-base experiments on GLUE tasks (SST2, MNLI)
- Comprehensive efficiency measurements
- LoRA baseline comparisons

### What's Excluded (per addendum)
- LLaMA model experiments
- Appendix-only experiments
- Specific baseline implementations (CoFi, Mask Tuning) - simplified comparisons used
- SQuAD and CNN/DM datasets (focus on classification tasks)

## Verification

The reproduction can be verified by:

1. **Performance**: APT should maintain ~98% accuracy vs LoRA while achieving significant sparsity
2. **Efficiency**: APT should show reduced memory usage and faster inference
3. **Trends**: Results should follow paper's reported trends across sparsity levels
4. **Ablations**: Different sparsity ratios should show expected accuracy/efficiency tradeoffs

## Technical Notes

- **GPU Requirements**: NVIDIA A10 GPU or equivalent recommended
- **Memory**: ~16GB GPU memory for RoBERTa-base experiments
- **Runtime**: ~2-4 hours for complete reproduction
- **Dependencies**: PyTorch, Transformers, Datasets (see requirements.txt)

## Citation

This reproduction is based on:
```
@inproceedings{zhao2024apt,
  title={APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference},
  author={Zhao, Bowen and Hajishirzi, Hannaneh and Cao, Qingqing},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```