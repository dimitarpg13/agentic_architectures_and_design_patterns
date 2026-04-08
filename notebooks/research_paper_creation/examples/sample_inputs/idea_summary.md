# Adaptive Sparse Attention for Long-Context Transformers

## Core Idea

We propose **AdaSparse**, an adaptive sparse attention mechanism that dynamically selects which tokens to attend to based on learned relevance scores, rather than using fixed sparsity patterns. Unlike prior sparse attention methods that use predetermined patterns (local windows, strided, or random), AdaSparse learns a lightweight token-scoring network that predicts attention importance in O(n) time, then computes full attention only over the top-k most relevant tokens.

## Key Components

1. **Token Relevance Predictor (TRP)**: A small MLP that takes token embeddings and produces per-token relevance scores. These scores determine which tokens each query should attend to.

2. **Adaptive Top-k Selection**: Instead of a fixed sparsity pattern, each query token selects its own set of k keys to attend to, where k is also adaptive based on the input complexity.

3. **Residual Dense Attention**: Every L-th layer uses full dense attention to prevent information loss from sparse approximation. This creates a "dense checkpoint" pattern.

## Theoretical Foundation

- The method achieves O(n·k) complexity where k << n, compared to O(n²) for dense attention
- We prove that with high probability, the top-k selection captures at least (1-ε) of the total attention mass when the attention distribution is sufficiently peaked
- The residual dense layers provide a theoretical guarantee of bounded approximation error across the full network depth

## Contributions

1. A novel adaptive sparse attention mechanism that learns input-dependent sparsity patterns
2. Theoretical analysis showing bounded approximation error
3. State-of-the-art results on long-context benchmarks (SCROLLS, LongBench) while using 3-5x less memory than dense attention
4. Comprehensive ablation studies demonstrating the importance of each component
