# Experimental Log — AdaSparse

## Setup

- **Base model**: Transformer decoder-only, 350M parameters, 24 layers
- **Training data**: C4 dataset, 100B tokens
- **Context lengths tested**: 4096, 8192, 16384, 32768
- **Hardware**: 8x A100 80GB GPUs
- **Training**: AdamW optimizer, lr=3e-4, cosine decay, 100k steps, batch size 256

## Baselines

| Method | Type | Complexity |
|--------|------|-----------|
| Dense Attention | Full | O(n²) |
| Longformer | Fixed sparse | O(n·w) |
| BigBird | Fixed sparse | O(n·(w+r+g)) |
| Performer | Linear approx | O(n·d) |
| FlashAttention-2 | Exact (IO-opt) | O(n²) |
| AdaSparse (ours) | Adaptive sparse | O(n·k) |

## Main Results — Perplexity (lower is better)

| Method | ctx=4096 | ctx=8192 | ctx=16384 | ctx=32768 |
|--------|----------|----------|-----------|-----------|
| Dense | 12.3 | 11.8 | 11.5 | OOM |
| Longformer | 13.1 | 12.5 | 12.1 | 11.9 |
| BigBird | 13.0 | 12.4 | 12.0 | 11.8 |
| Performer | 14.2 | 13.8 | 13.5 | 13.2 |
| FlashAttn-2 | 12.3 | 11.8 | 11.5 | 11.2 |
| AdaSparse | 12.5 | 11.9 | 11.6 | 11.3 |

## SCROLLS Benchmark (accuracy %)

| Method | QualityQA | NarrativeQA | Qasper | ContractNLI | Avg |
|--------|-----------|-------------|--------|-------------|-----|
| Longformer | 31.2 | 22.4 | 35.1 | 78.3 | 41.8 |
| BigBird | 32.1 | 23.1 | 36.2 | 79.1 | 42.6 |
| FlashAttn-2 | 35.4 | 26.3 | 39.8 | 82.7 | 46.1 |
| AdaSparse | 34.8 | 25.9 | 39.2 | 82.1 | 45.5 |

## Memory Usage (GB, batch=1)

| Method | ctx=4096 | ctx=8192 | ctx=16384 | ctx=32768 |
|--------|----------|----------|-----------|-----------|
| Dense | 4.2 | 14.8 | 58.1 | OOM |
| FlashAttn-2 | 2.1 | 4.0 | 8.1 | 16.2 |
| Longformer | 1.8 | 3.2 | 6.1 | 12.0 |
| AdaSparse | 1.5 | 2.8 | 5.2 | 10.1 |

## Ablation Studies

### Effect of k (sparsity budget) at ctx=16384

| k (% of n) | Perplexity | Memory (GB) | Throughput (tokens/s) |
|------------|-----------|-------------|----------------------|
| 5% | 12.1 | 3.8 | 42,100 |
| 10% | 11.8 | 4.5 | 38,200 |
| 20% | 11.6 | 5.2 | 31,500 |
| 50% | 11.5 | 8.4 | 18,900 |
| 100% (dense) | 11.5 | 58.1 | 6,200 |

### Component ablation at ctx=16384

| Configuration | Perplexity |
|--------------|-----------|
| Full AdaSparse | 11.6 |
| − Token Relevance Predictor (random top-k) | 12.0 |
| − Adaptive k (fixed k=10%) | 11.8 |
| − Residual dense layers | 11.9 |
| − All (random sparse, fixed k, no dense) | 12.4 |

## Training Efficiency

- AdaSparse adds ~5% overhead to training time vs Longformer at ctx=16384
- 2.3x faster training than dense attention at ctx=16384
- 4.1x faster than dense attention at ctx=32768 (dense OOMs without gradient checkpointing)
