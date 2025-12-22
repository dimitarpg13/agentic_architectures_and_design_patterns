# Agentic KPI Evaluation using Embeddings

A high-performance, embedding-based KPI evaluation system for agentic workflows that eliminates the need for expensive and non-deterministic LLM-as-scorer approaches.

## 🎯 Key Features

- **Embedding-based Metrics**: Uses semantic similarity for deterministic, fast evaluation
- **No LLM Required**: Eliminates API costs and latency of LLM-as-judge approaches
- **Three Core KPIs**:
  - **Accuracy**: Semantic similarity between output and ground truth
  - **Faithfulness**: Alignment between response and source context
  - **Relevance**: Query-response semantic alignment
- **Advanced Metrics**: Consistency, coverage, and specificity analysis
- **Production Ready**: Caching, batch processing, parallel evaluation
- **Multiple Providers**: Support for Sentence Transformers, OpenAI, Cohere, and custom embeddings

## 📊 Performance Comparison

| Approach | Speed | Cost | Deterministic | Offline |
|----------|-------|------|--------------|---------|
| **Embedding-based** | ~100 samples/sec | $0 (after model download) | ✅ Yes | ✅ Yes |
| LLM-as-Judge | ~0.5 samples/sec | ~$0.01/sample | ❌ No | ❌ No |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from agentic_kpi_embeddings import (
    AgenticKPIEvaluator,
    SentenceTransformerProvider,
    EvaluationSample,
    SimilarityMetric
)

# Initialize embedding provider
provider = SentenceTransformerProvider("all-MiniLM-L6-v2")

# Create evaluator
evaluator = AgenticKPIEvaluator(
    embedding_provider=provider,
    similarity_metric=SimilarityMetric.COSINE,
    use_cache=True
)

# Evaluate accuracy
accuracy = evaluator.calculate_accuracy(
    response="Paris is the capital of France.",
    ground_truth="The capital of France is Paris.",
    threshold=0.8
)
print(f"Accuracy: {accuracy.score:.4f}")

# Evaluate faithfulness
faithfulness = evaluator.calculate_faithfulness(
    response="Machine learning uses algorithms to learn from data.",
    context=[
        "ML is a subset of AI.",
        "Algorithms identify patterns in data."
    ]
)
print(f"Faithfulness: {faithfulness.score:.4f}")

# Evaluate relevance
relevance = evaluator.calculate_relevance(
    query="What is Python?",
    response="Python is a high-level programming language."
)
print(f"Relevance: {relevance.score:.4f}")
```

## 📈 Advanced Usage

### Batch Evaluation

```python
samples = [
    EvaluationSample(
        query="What is deep learning?",
        response="Deep learning uses neural networks with multiple layers.",
        context=["Neural networks process information.", "Deep learning is part of ML."],
        ground_truth="Deep learning is a subset of machine learning using neural networks."
    ),
    # ... more samples
]

# Evaluate batch with parallel processing
results_df = evaluator.evaluate_batch(samples, parallel=True, n_workers=4)

# Analyze results
analyzer = KPIAnalyzer(results_df)
summary = analyzer.get_summary_statistics()
correlations = analyzer.get_correlation_matrix()
```

### Production Pipeline

```python
class AgenticWorkflowPipeline:
    def __init__(self, embedding_provider, thresholds):
        self.evaluator = AgenticKPIEvaluator(
            embedding_provider=embedding_provider,
            similarity_metric=SimilarityMetric.COSINE,
            use_cache=True
        )
        self.thresholds = thresholds
    
    def evaluate_agent_response(self, query, response, context=None, ground_truth=None):
        # Evaluation logic with pass/fail criteria
        pass
```

## 🔧 Configuration Options

### Similarity Metrics
- `COSINE`: Cosine similarity (default, recommended)
- `EUCLIDEAN`: Normalized inverse Euclidean distance
- `DOT_PRODUCT`: Normalized dot product
- `ANGULAR`: Angular similarity
- `MANHATTAN`: Normalized inverse Manhattan distance

### Embedding Providers

#### Sentence Transformers (Recommended)
```python
provider = SentenceTransformerProvider("all-MiniLM-L6-v2")  # Fast, 384d
provider = SentenceTransformerProvider("all-mpnet-base-v2")  # Quality, 768d
```

#### OpenAI
```python
from agentic_kpi_embeddings import OpenAIProvider
provider = OpenAIProvider(api_key="your-key", model="text-embedding-3-small")
```

#### Custom Provider
```python
class CustomProvider(EmbeddingProvider):
    def embed(self, texts):
        # Your embedding logic
        pass
    
    def get_dimension(self):
        return 768
```

## 📊 Advanced Metrics

### Consistency Analysis
```python
from agentic_kpi_embeddings import AdvancedMetrics

consistency = AdvancedMetrics.calculate_consistency(
    responses=["Response 1", "Response 2", "Response 3"],
    embedding_provider=provider,
    similarity_calculator=SimilarityCalculator()
)
```

### Coverage Analysis
```python
coverage = AdvancedMetrics.calculate_coverage(
    response="Your agent's response",
    key_concepts=["concept1", "concept2", "concept3"],
    embedding_provider=provider,
    threshold=0.7
)
```

### Specificity Analysis
```python
specificity = AdvancedMetrics.calculate_specificity(
    response="Specific technical response",
    generic_responses=["This is interesting.", "There are many factors."],
    embedding_provider=provider
)
```

## 🔍 Architecture

```
AgenticKPIEvaluator
├── EmbeddingProvider (Abstract)
│   ├── SentenceTransformerProvider
│   ├── OpenAIProvider
│   └── CustomProvider
├── SimilarityCalculator
│   ├── cosine_similarity()
│   ├── euclidean_distance()
│   └── ...
├── EmbeddingCache
│   ├── Memory cache
│   └── Disk cache
└── Metrics
    ├── calculate_accuracy()
    ├── calculate_faithfulness()
    └── calculate_relevance()
```

## 🎯 Use Cases

1. **RAG System Evaluation**: Measure faithfulness to retrieved documents
2. **Code Generation**: Evaluate accuracy against reference implementations
3. **Question Answering**: Assess relevance of responses to queries
4. **Multi-Agent Systems**: Compare consistency across agent responses
5. **Summarization**: Measure coverage of key points
6. **Chatbot Quality**: Evaluate response specificity vs generic templates

## 📈 Benefits Over LLM-as-Judge

| Aspect | Embedding-based | LLM-as-Judge |
|--------|----------------|--------------|
| **Speed** | 100-200x faster | Slow (API calls) |
| **Cost** | Free after download | $0.01-0.10 per evaluation |
| **Determinism** | 100% reproducible | Variable outputs |
| **Offline** | Works without internet | Requires API access |
| **Parallelization** | Highly parallel | Limited by rate limits |
| **Bias** | Minimal | Model-dependent |

## 🔬 Technical Details

### Accuracy Metric
- Measures semantic similarity between response and ground truth
- Configurable threshold for binary pass/fail
- Uses normalized similarity scores (0-1 range)

### Faithfulness Metric
- Evaluates alignment with source context
- Supports multiple aggregation methods:
  - `max`: Maximum similarity to any context
  - `mean`: Average similarity across contexts
  - `weighted`: Position-weighted similarity

### Relevance Metric
- Direct query-response similarity
- Optional context-aware relevance
- Weighted combination of query and context alignment

## 📝 Export and Reporting

```python
# Export results
analyzer.export_report("report.json", format="json")
analyzer.export_report("results.csv", format="csv")
analyzer.export_report("report.html", format="html")

# Get summary statistics
summary = analyzer.get_summary_statistics()
correlations = analyzer.get_correlation_matrix()
outliers = analyzer.identify_outliers(threshold_std=2.0)
```

## 🚀 Performance Tips

1. **Enable Caching**: Reuse embeddings across evaluations
2. **Batch Processing**: Process multiple samples together
3. **Parallel Evaluation**: Use multiple workers for large datasets
4. **Choose Right Model**: Balance speed vs quality
   - Fast: `all-MiniLM-L6-v2` (384d)
   - Balanced: `all-mpnet-base-v2` (768d)
   - Quality: `all-roberta-large-v1` (1024d)

## 📚 Examples

See the included Jupyter notebook (`agentic_kpi_evaluation_demo.ipynb`) for comprehensive examples including:
- Basic and advanced metric calculations
- Batch evaluation workflows
- Production pipeline implementation
- Performance comparisons
- Visualization and analysis

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Additional embedding providers
- New similarity metrics
- Performance optimizations
- Additional KPI metrics
- Visualization improvements

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Sentence Transformers library for excellent embedding models
- The open-source community for continuous improvements

## 📧 Contact

For questions or suggestions, please open an issue on the project repository.

---

**Note**: This module provides deterministic, fast, and cost-effective evaluation of agentic workflows without requiring expensive LLM API calls. It's designed for production use with comprehensive caching, parallelization, and analysis capabilities.
