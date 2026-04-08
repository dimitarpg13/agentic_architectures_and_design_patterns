"""Web search tool abstraction with Tavily and mock implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class WebSearchTool(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Return a list of {title, url, content} dicts."""


class TavilySearchTool(WebSearchTool):
    def __init__(self, api_key: str):
        from tavily import TavilyClient
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        logger.info("Tavily search: %s (max %d)", query, max_results)
        response = self.client.search(query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]


class MockSearchTool(WebSearchTool):
    """Returns plausible mock results for demo/testing without API keys."""

    MOCK_PAPERS = [
        {
            "title": "Attention Is All You Need",
            "url": "https://arxiv.org/abs/1706.03762",
            "content": (
                "Vaswani et al. (2017) introduced the Transformer architecture, "
                "replacing recurrent layers with multi-head self-attention. "
                "The model achieves state-of-the-art results on machine translation "
                "with significantly reduced training time."
            ),
        },
        {
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "url": "https://arxiv.org/abs/1810.04805",
            "content": (
                "Devlin et al. (2019) proposed BERT, a bidirectional transformer "
                "pre-trained on masked language modeling and next sentence prediction. "
                "BERT achieves state-of-the-art results across 11 NLP benchmarks."
            ),
        },
        {
            "title": "Efficient Transformers: A Survey",
            "url": "https://arxiv.org/abs/2009.06732",
            "content": (
                "Tay et al. (2022) survey efficient transformer architectures "
                "that reduce the quadratic complexity of self-attention, including "
                "sparse attention, linear attention, and low-rank approximations."
            ),
        },
        {
            "title": "Longformer: The Long-Document Transformer",
            "url": "https://arxiv.org/abs/2004.05150",
            "content": (
                "Beltagy et al. (2020) propose Longformer with a combination of "
                "local windowed attention and task-specific global attention, scaling "
                "linearly with sequence length for documents up to 4096 tokens."
            ),
        },
        {
            "title": "FlashAttention: Fast and Memory-Efficient Exact Attention",
            "url": "https://arxiv.org/abs/2205.14135",
            "content": (
                "Dao et al. (2022) introduce FlashAttention, an IO-aware exact "
                "attention algorithm that reduces memory usage from quadratic to "
                "linear and achieves 2-4x wall-clock speedup over standard attention."
            ),
        },
        {
            "title": "Sparse Transformers: Generating Long Sequences with Sparse Attention",
            "url": "https://arxiv.org/abs/1904.10509",
            "content": (
                "Child et al. (2019) introduce sparse factorizations of the attention "
                "matrix, reducing compute from O(n^2) to O(n*sqrt(n)), enabling "
                "generation of sequences tens of thousands of tokens long."
            ),
        },
    ]

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        logger.info("Mock search: %s (returning %d results)", query, min(max_results, len(self.MOCK_PAPERS)))
        return self.MOCK_PAPERS[:max_results]


def create_search_tool(provider: str, **kwargs) -> WebSearchTool:
    if provider == "tavily":
        api_key = kwargs.get("api_key") or kwargs.get("tavily_api_key", "")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required for Tavily search")
        return TavilySearchTool(api_key=api_key)

    if provider == "mock":
        return MockSearchTool()

    raise ValueError(f"Unknown search provider: {provider}")
