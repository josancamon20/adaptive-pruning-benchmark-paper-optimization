#!/bin/bash

# APT Paper Reproduction Script
# Reproduces main experiments from "APT: Adaptive Pruning and Tuning Pretrained Language Models"

set -e  # Exit on error

echo "=== APT Paper Reproduction ==="
echo "Starting reproduction of main experiments..."

# Create necessary directories
mkdir -p results
mkdir -p baseline_results
mkdir -p comparison_results

# Install dependencies
echo "=== Installing Dependencies ==="
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Set CUDA if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected - enabling CUDA acceleration"
    export CUDA_VISIBLE_DEVICES=0
else
    echo "No GPU detected - using CPU"
fi

echo "=== Starting Experiments ==="

# Experiment 1: APT on RoBERTa-base with SST2 (60% sparsity)
echo "--- Experiment 1: APT RoBERTa-base SST2 (60% sparsity) ---"
python3 train_apt.py \
    --model_name roberta-base \
    --task_name sst2 \
    --target_sparsity 0.6 \
    --initial_rank 8 \
    --max_rank 64 \
    --learning_rate 2e-4 \
    --num_epochs 8 \
    --batch_size 16 \
    --output_dir ./results \
    --seed 42

echo "✓ Completed APT RoBERTa-base SST2"

# Experiment 2: APT on RoBERTa-base with MNLI (60% sparsity)  
echo "--- Experiment 2: APT RoBERTa-base MNLI (60% sparsity) ---"
python3 train_apt.py \
    --model_name roberta-base \
    --task_name mnli \
    --target_sparsity 0.6 \
    --initial_rank 8 \
    --max_rank 64 \
    --learning_rate 2e-4 \
    --num_epochs 6 \
    --batch_size 16 \
    --output_dir ./results \
    --seed 42

echo "✓ Completed APT RoBERTa-base MNLI"

# Experiment 3: LoRA Baseline on RoBERTa-base with SST2
echo "--- Experiment 3: LoRA Baseline RoBERTa-base SST2 ---"
python3 baseline_lora.py \
    --model_name roberta-base \
    --task_name sst2 \
    --rank 16 \
    --alpha 16.0 \
    --learning_rate 2e-4 \
    --num_epochs 8 \
    --batch_size 16 \
    --output_dir ./baseline_results

echo "✓ Completed LoRA Baseline SST2"

# Experiment 4: LoRA Baseline on RoBERTa-base with MNLI
echo "--- Experiment 4: LoRA Baseline RoBERTa-base MNLI ---"
python3 baseline_lora.py \
    --model_name roberta-base \
    --task_name mnli \
    --rank 16 \
    --alpha 16.0 \
    --learning_rate 2e-4 \
    --num_epochs 6 \
    --batch_size 16 \
    --output_dir ./baseline_results

echo "✓ Completed LoRA Baseline MNLI"

# Additional experiments with different sparsity levels
echo "--- Experiment 5: APT RoBERTa-base SST2 (40% sparsity) ---"
python3 train_apt.py \
    --model_name roberta-base \
    --task_name sst2 \
    --target_sparsity 0.4 \
    --initial_rank 8 \
    --max_rank 64 \
    --learning_rate 2e-4 \
    --num_epochs 8 \
    --batch_size 16 \
    --output_dir ./results \
    --seed 42

echo "✓ Completed APT RoBERTa-base SST2 (40% sparsity)"

echo "--- Experiment 6: APT RoBERTa-base SST2 (80% sparsity) ---" 
python3 train_apt.py \
    --model_name roberta-base \
    --task_name sst2 \
    --target_sparsity 0.8 \
    --initial_rank 8 \
    --max_rank 64 \
    --learning_rate 2e-4 \
    --num_epochs 8 \
    --batch_size 16 \
    --output_dir ./results \
    --seed 42

echo "✓ Completed APT RoBERTa-base SST2 (80% sparsity)"

# Generate comparison results
echo "=== Generating Comparison Results ==="
python3 -c "
import json
import os
from glob import glob
import numpy as np
import pandas as pd

# Collect all results
results = {}

# APT results
apt_files = glob('./results/metrics_*.json')
for file in apt_files:
    with open(file) as f:
        data = json.load(f)
        results[os.path.basename(file)] = data

# Baseline results  
baseline_files = glob('./baseline_results/*.json')
for file in baseline_files:
    with open(file) as f:
        data = json.load(f)
        results[os.path.basename(file)] = data

# Generate summary table
summary = []
for filename, data in results.items():
    if 'eval_accuracy' in data and len(data['eval_accuracy']) > 0:
        final_accuracy = data['eval_accuracy'][-1]
        final_memory = data['memory_gb'][-1] if 'memory_gb' in data else 0
        sparsity = data['sparsity_ratio'][-1] if 'sparsity_ratio' in data else 0
        throughput = data['throughput_samples_per_sec'][-1] if 'throughput_samples_per_sec' in data else 0
        
        summary.append({
            'experiment': filename,
            'accuracy': final_accuracy,
            'memory_gb': final_memory,
            'sparsity': sparsity,
            'throughput': throughput
        })

# Save summary
df = pd.DataFrame(summary)
df.to_csv('./comparison_results/experiment_summary.csv', index=False)
print('Summary saved to comparison_results/experiment_summary.csv')
print(df.to_string(index=False))

# Save detailed comparison
with open('./comparison_results/all_results.json', 'w') as f:
    json.dump(results, f, indent=2)
"

# Generate final report
echo "=== Generating Final Report ==="
python3 -c "
import json
import pandas as pd

# Load experiment summary
df = pd.read_csv('./comparison_results/experiment_summary.csv')

print('=== APT Paper Reproduction Results ===\\n')

apt_results = df[df['experiment'].str.contains('apt')]
lora_results = df[df['experiment'].str.contains('lora')]

print('APT Results:')
print(apt_results[['experiment', 'accuracy', 'sparsity', 'memory_gb', 'throughput']].to_string(index=False))
print()

print('LoRA Baseline Results:')  
print(lora_results[['experiment', 'accuracy', 'memory_gb', 'throughput']].to_string(index=False))
print()

# Performance comparison
if len(apt_results) > 0 and len(lora_results) > 0:
    apt_avg_acc = apt_results['accuracy'].mean()
    lora_avg_acc = lora_results['accuracy'].mean()
    apt_avg_mem = apt_results['memory_gb'].mean()  
    lora_avg_mem = lora_results['memory_gb'].mean()
    apt_avg_throughput = apt_results['throughput'].mean()
    lora_avg_throughput = lora_results['throughput'].mean()
    
    print('=== Comparison Summary ===')
    print(f'APT vs LoRA Accuracy: {apt_avg_acc:.3f} vs {lora_avg_acc:.3f}')
    print(f'APT vs LoRA Memory: {apt_avg_mem:.2f}GB vs {lora_avg_mem:.2f}GB')  
    print(f'APT vs LoRA Throughput: {apt_avg_throughput:.1f} vs {lora_avg_throughput:.1f} samples/sec')
    print(f'APT Sparsity: {apt_results[\"sparsity\"].mean():.1%}')
"

echo ""
echo "=== Reproduction Complete ==="
echo ""
echo "Results Summary:"
echo "- APT experiments completed with different sparsity levels (40%, 60%, 80%)"
echo "- LoRA baseline experiments completed for comparison"
echo "- Results saved in:"
echo "  * ./results/ (APT experiment results)"
echo "  * ./baseline_results/ (LoRA baseline results)"  
echo "  * ./comparison_results/ (comparative analysis)"
echo ""
echo "Key files generated:"
echo "- experiment_summary.csv: Summary of all experiments"
echo "- all_results.json: Detailed results from all runs"
echo "- Individual metric files for each experiment"
echo ""

# Verify GPU memory usage was tracked
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Memory Usage Summary:"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
fi

echo "Reproduction script completed successfully!"