"""Tests for the web search tool abstraction."""

import pytest

from tools.web_search import MockSearchTool, create_search_tool


class TestMockSearchTool:

    def test_returns_results(self):
        tool = MockSearchTool()
        results = tool.search("transformers")
        assert len(results) > 0
        assert all("title" in r for r in results)
        assert all("url" in r for r in results)
        assert all("content" in r for r in results)

    def test_respects_max_results(self):
        tool = MockSearchTool()
        results = tool.search("query", max_results=2)
        assert len(results) == 2


class TestCreateSearchTool:

    def test_creates_mock_tool(self):
        tool = create_search_tool("mock")
        assert isinstance(tool, MockSearchTool)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown search provider"):
            create_search_tool("nonexistent")

    def test_tavily_without_key_raises(self):
        with pytest.raises(ValueError, match="TAVILY_API_KEY"):
            create_search_tool("tavily", api_key="")
