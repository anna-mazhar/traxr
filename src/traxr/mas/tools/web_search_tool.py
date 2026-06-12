"""Web search tool for finding information online.

Supports multiple backends:
- DuckDuckGo (free, no API key required)
- Serper (optional, requires SERPER_API_KEY)
- Tavily (optional, requires TAVILY_API_KEY)
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus
from .base import BaseTool, ToolResult

# Optional imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class WebSearchTool(BaseTool):
    """Tool for searching the web.

    Provides operations:
    - search: Search the web for a query
    - search_news: Search for recent news
    """

    def __init__(
        self,
        backend: str = "duckduckgo",
        api_key: Optional[str] = None,
        max_results: int = 5,
    ):
        """Initialize WebSearchTool.

        Args:
            backend: Search backend to use ('duckduckgo', 'serper', 'tavily')
            api_key: API key for paid backends (or set via environment variable)
            max_results: Maximum number of results to return
        """
        super().__init__(name="web_search")
        self.backend = backend.lower()
        self.max_results = max_results

        # Get API key from parameter or environment
        if self.backend == "serper":
            self.api_key = api_key or os.environ.get("SERPER_API_KEY")
        elif self.backend == "tavily":
            self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        else:
            self.api_key = None

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute web search operation."""
        if not HAS_REQUESTS:
            return ToolResult(
                success=False,
                output=None,
                error="requests library required. Install with: pip install requests"
            )

        operations = {
            "search": self._search,
            "search_news": self._search_news,
        }

        if operation not in operations:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown operation '{operation}'. Available: {list(operations.keys())}"
            )

        try:
            return operations[operation](**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error in {operation}: {str(e)}"
            )

    def get_available_operations(self) -> List[str]:
        """Get list of available operations."""
        return ["search", "search_news"]

    def get_schema(self) -> "ToolSchema":
        from .tool_schema import ToolSchema, OperationSchema, ToolParameterSchema
        return ToolSchema(
            name="web_search",
            description="Search the web for information. Returns titles, URLs, and snippets.",
            operations={
                "search": OperationSchema(
                    name="search",
                    description="Search the web for a query. Returns a list of results with title, URL, and snippet.",
                    parameters=[
                        ToolParameterSchema(name="query", type="string", description="Search query string"),
                        ToolParameterSchema(name="max_results", type="integer", description="Maximum number of results to return", required=False, default=5),
                    ],
                ),
                "search_news": OperationSchema(
                    name="search_news",
                    description="Search for recent news articles matching a query.",
                    parameters=[
                        ToolParameterSchema(name="query", type="string", description="News search query string"),
                        ToolParameterSchema(name="max_results", type="integer", description="Maximum number of results", required=False, default=5),
                    ],
                ),
            },
        )

    def _search(self, query: str, max_results: Optional[int] = None) -> ToolResult:
        """Search the web for a query.

        Args:
            query: Search query string
            max_results: Override default max results

        Returns:
            ToolResult with list of search results
        """
        max_results = max_results or self.max_results

        if self.backend == "duckduckgo":
            return self._search_duckduckgo(query, max_results)
        elif self.backend == "serper":
            return self._search_serper(query, max_results)
        elif self.backend == "tavily":
            return self._search_tavily(query, max_results)
        else:
            return ToolResult(
                success=False,
                output=None,
                error=f"Unknown backend: {self.backend}"
            )

    def _search_news(self, query: str, max_results: Optional[int] = None) -> ToolResult:
        """Search for recent news.

        Args:
            query: Search query string
            max_results: Override default max results

        Returns:
            ToolResult with list of news results
        """
        max_results = max_results or self.max_results

        if self.backend == "serper":
            return self._search_serper(query, max_results, search_type="news")
        else:
            # Fallback: add "news" to query
            return self._search(f"{query} news", max_results)

    def _search_duckduckgo(self, query: str, max_results: int) -> ToolResult:
        """Search using DuckDuckGo (free, no API key).

        Uses the DuckDuckGo HTML interface and parses results.
        """
        try:
            # Use DuckDuckGo HTML search
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parse results from HTML
            results = self._parse_duckduckgo_html(response.text, max_results)

            if not results:
                return ToolResult(
                    success=True,
                    output=[],
                    metadata={"query": query, "backend": "duckduckgo", "count": 0}
                )

            return ToolResult(
                success=True,
                output=results,
                metadata={
                    "query": query,
                    "backend": "duckduckgo",
                    "count": len(results),
                }
            )

        except requests.RequestException as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"DuckDuckGo search failed: {str(e)}"
            )

    def _parse_duckduckgo_html(self, html: str, max_results: int) -> List[Dict[str, str]]:
        """Parse DuckDuckGo HTML results."""
        results = []

        # Find result blocks - DuckDuckGo uses class="result__a" for links
        # and class="result__snippet" for snippets
        link_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
        snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]*(?:<[^>]*>[^<]*)*)</a>'

        links = re.findall(link_pattern, html)
        snippets = re.findall(snippet_pattern, html)

        for i, (url, title) in enumerate(links[:max_results]):
            snippet = ""
            if i < len(snippets):
                # Clean HTML tags from snippet
                snippet = re.sub(r'<[^>]+>', '', snippets[i])
                snippet = snippet.strip()

            # Clean up the URL (DuckDuckGo uses redirect URLs)
            if "uddg=" in url:
                # Extract actual URL from DuckDuckGo redirect
                match = re.search(r'uddg=([^&]+)', url)
                if match:
                    from urllib.parse import unquote
                    url = unquote(match.group(1))

            results.append({
                "title": title.strip(),
                "url": url,
                "snippet": snippet,
            })

        return results

    def _search_serper(
        self,
        query: str,
        max_results: int,
        search_type: str = "search"
    ) -> ToolResult:
        """Search using Serper API (requires API key).

        Get API key at: https://serper.dev/
        """
        if not self.api_key:
            return ToolResult(
                success=False,
                output=None,
                error="Serper API key required. Set SERPER_API_KEY environment variable."
            )

        try:
            url = f"https://google.serper.dev/{search_type}"
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "q": query,
                "num": max_results,
            }

            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Parse results
            results = []

            # Handle organic results
            for item in data.get("organic", [])[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                })

            # Handle news results
            for item in data.get("news", [])[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "date": item.get("date", ""),
                })

            # Include knowledge graph if available
            knowledge_graph = data.get("knowledgeGraph")
            if knowledge_graph:
                results.insert(0, {
                    "title": knowledge_graph.get("title", ""),
                    "url": knowledge_graph.get("website", ""),
                    "snippet": knowledge_graph.get("description", ""),
                    "type": "knowledge_graph",
                })

            return ToolResult(
                success=True,
                output=results[:max_results],
                metadata={
                    "query": query,
                    "backend": "serper",
                    "count": len(results[:max_results]),
                    "search_type": search_type,
                }
            )

        except requests.RequestException as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Serper search failed: {str(e)}"
            )

    def _search_tavily(self, query: str, max_results: int) -> ToolResult:
        """Search using Tavily API (AI-optimized, requires API key).

        Get API key at: https://tavily.com/
        """
        if not self.api_key:
            return ToolResult(
                success=False,
                output=None,
                error="Tavily API key required. Set TAVILY_API_KEY environment variable."
            )

        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": True,
                "include_raw_content": False,
            }

            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            data = response.json()

            results = []

            # Include AI-generated answer if available
            if data.get("answer"):
                results.append({
                    "title": "AI Summary",
                    "url": "",
                    "snippet": data["answer"],
                    "type": "ai_answer",
                })

            # Add search results
            for item in data.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "score": item.get("score", 0),
                })

            return ToolResult(
                success=True,
                output=results[:max_results + 1],  # +1 for AI answer
                metadata={
                    "query": query,
                    "backend": "tavily",
                    "count": len(results),
                    "has_ai_answer": bool(data.get("answer")),
                }
            )

        except requests.RequestException as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tavily search failed: {str(e)}"
            )
