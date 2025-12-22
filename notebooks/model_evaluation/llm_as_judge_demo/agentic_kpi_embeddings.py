"""
Agentic Workflow KPI Evaluation using Embeddings

This module provides embedding-based evaluation metrics for agentic workflows,
focusing on accuracy, faithfulness, and relevance without using LLM-as-scorer approaches.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import pickle
from abc import ABC, abstractmethod
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr, spearmanr
import logging
from datetime import datetime
from functools import lru_cache
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Supported similarity metrics for embedding comparison."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    ANGULAR = "angular"
    MANHATTAN = "manhattan"


class EmbeddingModel(Enum):
    """Supported embedding models."""
    SENTENCE_TRANSFORMER = "sentence-transformer"
    OPENAI = "openai"
    COHERE = "cohere"
    CUSTOM = "custom"


@dataclass
class KPIResult:
    """Container for KPI evaluation results."""
    metric_name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class EvaluationSample:
    """Container for a single evaluation sample."""
    query: str
    response: str
    context: Optional[List[str]] = None
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for input texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of embeddings."""
        pass


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformer embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using Sentence Transformers."""
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True)
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
            self._dimension = 1536 if "3-small" in model else 3072
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        if isinstance(texts, str):
            texts = [texts]
        
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self._dimension


class EmbeddingCache:
    """Cache for storing computed embeddings."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".agentic_kpi_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
    
    def _get_key(self, text: str, model_id: str) -> str:
        """Generate cache key for text and model combination."""
        content = f"{model_id}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, model_id: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache."""
        key = self._get_key(text, model_id)
        
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                embedding = pickle.load(f)
                self.memory_cache[key] = embedding
                return embedding
        
        return None
    
    def set(self, text: str, model_id: str, embedding: np.ndarray):
        """Store embedding in cache."""
        key = self._get_key(text, model_id)
        
        # Store in memory
        self.memory_cache[key] = embedding
        
        # Store on disk
        cache_file = self.cache_dir / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)


class SimilarityCalculator:
    """Calculate similarity between embeddings using various metrics."""
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return 1 - cosine(emb1, emb2)
    
    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate normalized inverse euclidean distance."""
        distance = euclidean(emb1, emb2)
        # Normalize to 0-1 range (1 being most similar)
        return 1 / (1 + distance)
    
    @staticmethod
    def dot_product(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate normalized dot product."""
        product = np.dot(emb1, emb2)
        # Normalize by magnitudes
        norm_product = product / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return (norm_product + 1) / 2  # Scale to 0-1
    
    @staticmethod
    def angular_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate angular similarity (1 - angular distance)."""
        cos_sim = SimilarityCalculator.cosine_similarity(emb1, emb2)
        angle = np.arccos(np.clip(cos_sim, -1, 1)) / np.pi
        return 1 - angle
    
    @staticmethod
    def manhattan_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate normalized inverse Manhattan distance."""
        distance = np.sum(np.abs(emb1 - emb2))
        # Normalize to 0-1 range
        max_distance = len(emb1) * 2  # Maximum possible Manhattan distance
        return 1 - (distance / max_distance)
    
    def calculate(self, 
                  emb1: np.ndarray, 
                  emb2: np.ndarray, 
                  metric: SimilarityMetric = SimilarityMetric.COSINE) -> float:
        """Calculate similarity using specified metric."""
        metric_map = {
            SimilarityMetric.COSINE: self.cosine_similarity,
            SimilarityMetric.EUCLIDEAN: self.euclidean_distance,
            SimilarityMetric.DOT_PRODUCT: self.dot_product,
            SimilarityMetric.ANGULAR: self.angular_distance,
            SimilarityMetric.MANHATTAN: self.manhattan_distance
        }
        
        return metric_map[metric](emb1, emb2)


class AgenticKPIEvaluator:
    """Main evaluator class for agentic workflow KPIs using embeddings."""
    
    def __init__(self, 
                 embedding_provider: EmbeddingProvider,
                 similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
                 use_cache: bool = True,
                 cache_dir: Optional[Path] = None,
                 batch_size: int = 32):
        """
        Initialize the KPI evaluator.
        
        Args:
            embedding_provider: Provider for generating embeddings
            similarity_metric: Metric to use for similarity calculations
            use_cache: Whether to cache embeddings
            cache_dir: Directory for caching embeddings
            batch_size: Batch size for embedding generation
        """
        self.embedding_provider = embedding_provider
        self.similarity_metric = similarity_metric
        self.similarity_calculator = SimilarityCalculator()
        self.batch_size = batch_size
        
        self.cache = EmbeddingCache(cache_dir) if use_cache else None
        self.model_id = str(type(embedding_provider).__name__)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache if available."""
        if self.cache:
            cached = self.cache.get(text, self.model_id)
            if cached is not None:
                return cached
        
        embedding = self.embedding_provider.embed(text)
        if len(embedding.shape) > 1:
            embedding = embedding[0]
        
        if self.cache:
            self.cache.set(text, self.model_id, embedding)
        
        return embedding
    
    def _batch_embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings in batches for efficiency."""
        embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for text in batch:
                if self.cache:
                    cached = self.cache.get(text, self.model_id)
                    if cached is not None:
                        batch_embeddings.append(cached)
                    else:
                        emb = self.embedding_provider.embed(text)
                        if len(emb.shape) > 1:
                            emb = emb[0]
                        self.cache.set(text, self.model_id, emb)
                        batch_embeddings.append(emb)
                else:
                    emb = self.embedding_provider.embed(text)
                    if len(emb.shape) > 1:
                        emb = emb[0]
                    batch_embeddings.append(emb)
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def calculate_accuracy(self, 
                          response: str, 
                          ground_truth: str,
                          threshold: float = 0.8) -> KPIResult:
        """
        Calculate accuracy using semantic similarity between response and ground truth.
        
        Args:
            response: Agent's response
            ground_truth: Expected/correct response
            threshold: Similarity threshold for binary accuracy
        
        Returns:
            KPIResult with accuracy score
        """
        response_emb = self._get_embedding(response)
        truth_emb = self._get_embedding(ground_truth)
        
        similarity = self.similarity_calculator.calculate(
            response_emb, truth_emb, self.similarity_metric
        )
        
        # Calculate binary accuracy based on threshold
        binary_accuracy = 1.0 if similarity >= threshold else 0.0
        
        return KPIResult(
            metric_name="accuracy",
            score=similarity,
            details={
                "binary_accuracy": binary_accuracy,
                "threshold": threshold,
                "similarity_metric": self.similarity_metric.value,
                "raw_similarity": similarity
            }
        )
    
    def calculate_faithfulness(self, 
                              response: str, 
                              context: List[str],
                              aggregation: str = "max") -> KPIResult:
        """
        Calculate faithfulness by measuring how well the response aligns with context.
        
        Args:
            response: Agent's response
            context: List of context documents/passages
            aggregation: How to aggregate similarities ("max", "mean", "weighted")
        
        Returns:
            KPIResult with faithfulness score
        """
        if not context:
            return KPIResult(
                metric_name="faithfulness",
                score=0.0,
                details={"error": "No context provided"}
            )
        
        response_emb = self._get_embedding(response)
        context_embs = self._batch_embed(context)
        
        # Calculate similarity with each context piece
        similarities = []
        for ctx_emb in context_embs:
            sim = self.similarity_calculator.calculate(
                response_emb, ctx_emb, self.similarity_metric
            )
            similarities.append(sim)
        
        # Aggregate similarities
        if aggregation == "max":
            score = max(similarities)
        elif aggregation == "mean":
            score = np.mean(similarities)
        elif aggregation == "weighted":
            # Weight by position (earlier context more important)
            weights = np.array([1 / (i + 1) for i in range(len(similarities))])
            weights = weights / weights.sum()
            score = np.average(similarities, weights=weights)
        else:
            score = np.mean(similarities)
        
        return KPIResult(
            metric_name="faithfulness",
            score=score,
            details={
                "individual_similarities": similarities,
                "aggregation_method": aggregation,
                "num_context_pieces": len(context),
                "max_similarity": max(similarities),
                "min_similarity": min(similarities),
                "std_similarity": np.std(similarities)
            }
        )
    
    def calculate_relevance(self, 
                           query: str, 
                           response: str,
                           context: Optional[List[str]] = None) -> KPIResult:
        """
        Calculate relevance of response to the query.
        
        Args:
            query: User query/question
            response: Agent's response
            context: Optional context to consider
        
        Returns:
            KPIResult with relevance score
        """
        query_emb = self._get_embedding(query)
        response_emb = self._get_embedding(response)
        
        # Direct query-response relevance
        query_response_sim = self.similarity_calculator.calculate(
            query_emb, response_emb, self.similarity_metric
        )
        
        details = {
            "query_response_similarity": query_response_sim
        }
        
        # If context provided, calculate context-aware relevance
        if context:
            context_embs = self._batch_embed(context)
            
            # Check if response is relevant to both query and context
            context_similarities = []
            for ctx_emb in context_embs:
                ctx_sim = self.similarity_calculator.calculate(
                    response_emb, ctx_emb, self.similarity_metric
                )
                context_similarities.append(ctx_sim)
            
            avg_context_sim = np.mean(context_similarities)
            
            # Combined relevance score
            score = 0.7 * query_response_sim + 0.3 * avg_context_sim
            
            details.update({
                "context_similarity": avg_context_sim,
                "combined_score": score,
                "weight_query": 0.7,
                "weight_context": 0.3
            })
        else:
            score = query_response_sim
        
        return KPIResult(
            metric_name="relevance",
            score=score,
            details=details
        )
    
    def evaluate_sample(self, sample: EvaluationSample) -> Dict[str, KPIResult]:
        """
        Evaluate all KPIs for a single sample.
        
        Args:
            sample: Evaluation sample containing query, response, context, etc.
        
        Returns:
            Dictionary mapping metric names to results
        """
        results = {}
        
        # Calculate relevance
        results["relevance"] = self.calculate_relevance(
            sample.query, 
            sample.response,
            sample.context
        )
        
        # Calculate faithfulness if context available
        if sample.context:
            results["faithfulness"] = self.calculate_faithfulness(
                sample.response,
                sample.context
            )
        
        # Calculate accuracy if ground truth available
        if sample.ground_truth:
            results["accuracy"] = self.calculate_accuracy(
                sample.response,
                sample.ground_truth
            )
        
        return results
    
    def evaluate_batch(self, 
                      samples: List[EvaluationSample],
                      parallel: bool = True,
                      n_workers: int = 4) -> pd.DataFrame:
        """
        Evaluate multiple samples and return aggregated results.
        
        Args:
            samples: List of evaluation samples
            parallel: Whether to process samples in parallel
            n_workers: Number of parallel workers
        
        Returns:
            DataFrame with evaluation results
        """
        all_results = []
        
        if parallel:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(self.evaluate_sample, sample): i 
                    for i, sample in enumerate(samples)
                }
                
                for future in as_completed(futures):
                    sample_idx = futures[future]
                    try:
                        results = future.result()
                        for metric_name, result in results.items():
                            all_results.append({
                                "sample_idx": sample_idx,
                                "metric": metric_name,
                                "score": result.score,
                                **result.details
                            })
                    except Exception as e:
                        logger.error(f"Error evaluating sample {sample_idx}: {e}")
        else:
            for i, sample in enumerate(samples):
                results = self.evaluate_sample(sample)
                for metric_name, result in results.items():
                    all_results.append({
                        "sample_idx": i,
                        "metric": metric_name,
                        "score": result.score,
                        **result.details
                    })
        
        return pd.DataFrame(all_results)


class AdvancedMetrics:
    """Advanced metrics for more sophisticated evaluation."""
    
    @staticmethod
    def calculate_consistency(responses: List[str], 
                            embedding_provider: EmbeddingProvider,
                            similarity_calculator: SimilarityCalculator,
                            metric: SimilarityMetric = SimilarityMetric.COSINE) -> float:
        """
        Calculate consistency across multiple responses.
        
        Args:
            responses: List of responses to check consistency
            embedding_provider: Provider for embeddings
            similarity_calculator: Calculator for similarities
            metric: Similarity metric to use
        
        Returns:
            Consistency score (0-1)
        """
        if len(responses) < 2:
            return 1.0
        
        embeddings = [embedding_provider.embed(resp)[0] if len(embedding_provider.embed(resp).shape) > 1 
                     else embedding_provider.embed(resp) for resp in responses]
        
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = similarity_calculator.calculate(
                    embeddings[i], embeddings[j], metric
                )
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    @staticmethod
    def calculate_coverage(response: str,
                          key_concepts: List[str],
                          embedding_provider: EmbeddingProvider,
                          similarity_calculator: SimilarityCalculator,
                          metric: SimilarityMetric = SimilarityMetric.COSINE,
                          threshold: float = 0.7) -> float:
        """
        Calculate how well the response covers key concepts.
        
        Args:
            response: Response text
            key_concepts: List of key concepts to check
            embedding_provider: Provider for embeddings
            similarity_calculator: Calculator for similarities
            metric: Similarity metric to use
            threshold: Similarity threshold for concept coverage
        
        Returns:
            Coverage score (0-1)
        """
        if not key_concepts:
            return 1.0
        
        response_emb = embedding_provider.embed(response)
        if len(response_emb.shape) > 1:
            response_emb = response_emb[0]
        
        covered = 0
        for concept in key_concepts:
            concept_emb = embedding_provider.embed(concept)
            if len(concept_emb.shape) > 1:
                concept_emb = concept_emb[0]
            
            similarity = similarity_calculator.calculate(
                response_emb, concept_emb, metric
            )
            
            if similarity >= threshold:
                covered += 1
        
        return covered / len(key_concepts)
    
    @staticmethod
    def calculate_specificity(response: str,
                            generic_responses: List[str],
                            embedding_provider: EmbeddingProvider,
                            similarity_calculator: SimilarityCalculator,
                            metric: SimilarityMetric = SimilarityMetric.COSINE) -> float:
        """
        Calculate how specific (non-generic) a response is.
        
        Args:
            response: Response to evaluate
            generic_responses: List of generic response templates
            embedding_provider: Provider for embeddings
            similarity_calculator: Calculator for similarities
            metric: Similarity metric to use
        
        Returns:
            Specificity score (0-1, higher is more specific)
        """
        if not generic_responses:
            return 1.0
        
        response_emb = embedding_provider.embed(response)
        if len(response_emb.shape) > 1:
            response_emb = response_emb[0]
        
        similarities = []
        for generic in generic_responses:
            generic_emb = embedding_provider.embed(generic)
            if len(generic_emb.shape) > 1:
                generic_emb = generic_emb[0]
            
            similarity = similarity_calculator.calculate(
                response_emb, generic_emb, metric
            )
            similarities.append(similarity)
        
        # Lower similarity to generic responses means higher specificity
        avg_generic_similarity = np.mean(similarities)
        return 1 - avg_generic_similarity


class KPIAnalyzer:
    """Analyze and visualize KPI results."""
    
    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize analyzer with results DataFrame.
        
        Args:
            results_df: DataFrame containing evaluation results
        """
        self.results_df = results_df
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for all metrics."""
        summary = self.results_df.groupby('metric')['score'].agg([
            'mean', 'std', 'min', 'max', 'median',
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75))
        ]).round(4)
        
        return summary
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation between different metrics."""
        pivot_df = self.results_df.pivot_table(
            values='score', 
            index='sample_idx', 
            columns='metric'
        )
        
        return pivot_df.corr()
    
    def identify_outliers(self, threshold_std: float = 2.0) -> pd.DataFrame:
        """Identify outlier samples based on standard deviation."""
        outliers = []
        
        for metric in self.results_df['metric'].unique():
            metric_data = self.results_df[self.results_df['metric'] == metric]
            mean_score = metric_data['score'].mean()
            std_score = metric_data['score'].std()
            
            threshold_low = mean_score - threshold_std * std_score
            threshold_high = mean_score + threshold_std * std_score
            
            outlier_samples = metric_data[
                (metric_data['score'] < threshold_low) | 
                (metric_data['score'] > threshold_high)
            ]
            
            for _, row in outlier_samples.iterrows():
                outliers.append({
                    'sample_idx': row['sample_idx'],
                    'metric': metric,
                    'score': row['score'],
                    'deviation': (row['score'] - mean_score) / std_score
                })
        
        return pd.DataFrame(outliers)
    
    def export_report(self, filepath: Path, format: str = "json"):
        """
        Export analysis report to file.
        
        Args:
            filepath: Path to save the report
            format: Format for export ("json", "csv", "html")
        """
        report = {
            "summary": self.get_summary_statistics().to_dict(),
            "correlations": self.get_correlation_matrix().to_dict(),
            "outliers": self.identify_outliers().to_dict(),
            "raw_results": self.results_df.to_dict()
        }
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        elif format == "csv":
            self.results_df.to_csv(filepath, index=False)
        elif format == "html":
            html = self.results_df.to_html()
            with open(filepath, 'w') as f:
                f.write(html)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Example usage and testing
if __name__ == "__main__":
    # Example: Initialize with Sentence Transformers
    print("Initializing Agentic KPI Evaluator...")
    
    # Create embedding provider
    try:
        provider = SentenceTransformerProvider("all-MiniLM-L6-v2")
        print("✓ Embedding provider initialized")
    except ImportError:
        print("Note: Install sentence-transformers for full functionality")
        print("Using mock provider for demonstration")
        
        class MockProvider(EmbeddingProvider):
            def embed(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                return np.random.randn(len(texts), 384)
            def get_dimension(self):
                return 384
        
        provider = MockProvider()
    
    # Create evaluator
    evaluator = AgenticKPIEvaluator(
        embedding_provider=provider,
        similarity_metric=SimilarityMetric.COSINE,
        use_cache=True
    )
    
    # Example evaluation
    sample = EvaluationSample(
        query="What is the capital of France?",
        response="The capital of France is Paris, which is located in the north-central part of the country.",
        context=["France is a country in Western Europe.", "Paris is the largest city in France."],
        ground_truth="Paris is the capital of France."
    )
    
    print("\nEvaluating sample...")
    results = evaluator.evaluate_sample(sample)
    
    for metric_name, result in results.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Score: {result.score:.4f}")
        print(f"  Details: {result.details}")
    
    # Batch evaluation example
    samples = [
        EvaluationSample(
            query="What is machine learning?",
            response="Machine learning is a type of artificial intelligence that enables systems to learn from data.",
            context=["ML is a subset of AI.", "It uses algorithms to identify patterns."],
            ground_truth="Machine learning is a branch of AI that uses data to improve performance."
        ),
        EvaluationSample(
            query="Explain quantum computing",
            response="Quantum computing uses quantum bits or qubits to process information.",
            context=["Quantum computers leverage quantum mechanics.", "Qubits can exist in superposition."],
            ground_truth="Quantum computing is a type of computation using quantum-mechanical phenomena."
        )
    ]
    
    print("\n\nBatch Evaluation:")
    results_df = evaluator.evaluate_batch(samples, parallel=False)
    
    # Analyze results
    analyzer = KPIAnalyzer(results_df)
    print("\nSummary Statistics:")
    print(analyzer.get_summary_statistics())
    
    print("\nMetric Correlations:")
    print(analyzer.get_correlation_matrix())
    
    print("\n✅ Agentic KPI Evaluator module ready for use!")
