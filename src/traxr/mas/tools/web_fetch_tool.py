"""Web fetch tool for retrieving and parsing web page content.

Extracts clean text content from URLs, handling various page types.
"""

import os
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urljoin
from .base import BaseTool, ToolResult

# Optional imports
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class WebFetchTool(BaseTool):
    """Tool for fetching and parsing web page content.

    Provides operations:
    - fetch: Fetch and extract text content from a URL
    - fetch_raw: Fetch raw HTML from a URL
    - extract_links: Extract all links from a page
    - extract_tables: Extract tables from a page
    """

    def __init__(
        self,
        timeout: int = 15,
        max_content_length: int = 50000,
        user_agent: Optional[str] = None,
    ):
        """Initialize WebFetchTool.

        Args:
            timeout: Request timeout in seconds
            max_content_length: Maximum content length to return (chars)
            user_agent: Custom user agent string
        """
        super().__init__(name="web_fetch")
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self._session = None

    def _get_session(self) -> requests.Session:
        """Get or create requests session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            })
        return self._session

    def close(self) -> None:
        """Close the requests session to release connections."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    def __del__(self):
        """Cleanup on garbage collection."""
        self.close()

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """Execute web fetch operation."""
        if not HAS_REQUESTS:
            return ToolResult(
                success=False,
                output=None,
                error="requests library required. Install with: pip install requests"
            )

        operations = {
            "fetch": self._fetch,
            "fetch_raw": self._fetch_raw,
            "extract_links": self._extract_links,
            "extract_tables": self._extract_tables,
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
        return ["fetch", "fetch_raw", "extract_links", "extract_tables"]

    def get_schema(self) -> "ToolSchema":
        from .tool_schema import ToolSchema, OperationSchema, ToolParameterSchema
        return ToolSchema(
            name="web_fetch",
            description="Fetch and parse web page content. Extracts clean text, links, and tables from URLs.",
            operations={
                "fetch": OperationSchema(
                    name="fetch",
                    description="Fetch a URL and extract clean text content. Removes scripts, styles, and navigation. Returns title and main content.",
                    parameters=[
                        ToolParameterSchema(name="url", type="string", description="URL to fetch"),
                        ToolParameterSchema(name="extract_main", type="boolean", description="If true, try to extract only main content area", required=False, default=True),
                    ],
                ),
                "fetch_raw": OperationSchema(
                    name="fetch_raw",
                    description="Fetch raw HTML from a URL without processing.",
                    parameters=[
                        ToolParameterSchema(name="url", type="string", description="URL to fetch"),
                    ],
                ),
                "extract_links": OperationSchema(
                    name="extract_links",
                    description="Extract all hyperlinks from a web page.",
                    parameters=[
                        ToolParameterSchema(name="url", type="string", description="URL to extract links from"),
                        ToolParameterSchema(name="absolute", type="boolean", description="Convert relative links to absolute URLs", required=False, default=True),
                    ],
                ),
                "extract_tables": OperationSchema(
                    name="extract_tables",
                    description="Extract HTML tables from a web page as lists of rows.",
                    parameters=[
                        ToolParameterSchema(name="url", type="string", description="URL to extract tables from"),
                    ],
                ),
            },
        )

    def _fetch(self, url: str, extract_main: bool = True) -> ToolResult:
        """Fetch and extract clean text content from a URL.

        Args:
            url: URL to fetch
            extract_main: If True, try to extract only main content

        Returns:
            ToolResult with extracted text content
        """
        if not HAS_BS4:
            return ToolResult(
                success=False,
                output=None,
                error="beautifulsoup4 required. Install with: pip install beautifulsoup4"
            )

        try:
            session = self._get_session()
            response = session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                # Not HTML - return raw text if it's text-like
                if "text/" in content_type or "application/json" in content_type:
                    content = response.text[:self.max_content_length]
                    return ToolResult(
                        success=True,
                        output=content,
                        metadata={
                            "url": url,
                            "content_type": content_type,
                            "length": len(content),
                        }
                    )
                else:
                    return ToolResult(
                        success=False,
                        output=None,
                        error=f"Cannot extract text from content type: {content_type}"
                    )

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header",
                                "aside", "noscript", "iframe", "form"]):
                element.decompose()

            # Try to find main content
            if extract_main:
                content = self._extract_main_content(soup)
            else:
                content = soup.get_text(separator="\n", strip=True)

            # Clean up whitespace
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = content.strip()

            # Truncate if needed
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "\n... [truncated]"

            # Extract title
            title = ""
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            return ToolResult(
                success=True,
                output={
                    "title": title,
                    "content": content,
                    "url": url,
                },
                metadata={
                    "url": url,
                    "title": title,
                    "content_length": len(content),
                }
            )

        except requests.RequestException as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to fetch URL: {str(e)}"
            )

    def _extract_main_content(self, soup) -> str:
        """Try to extract main content from page."""
        # Try common main content containers
        main_selectors = [
            "main",
            "article",
            '[role="main"]',
            "#content",
            "#main",
            ".content",
            ".main",
            ".post-content",
            ".article-content",
            ".entry-content",
        ]

        for selector in main_selectors:
            main = soup.select_one(selector)
            if main and len(main.get_text(strip=True)) > 200:
                return main.get_text(separator="\n", strip=True)

        # Fallback to body
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)

        return soup.get_text(separator="\n", strip=True)

    def _fetch_raw(self, url: str) -> ToolResult:
        """Fetch raw HTML from a URL.

        Args:
            url: URL to fetch

        Returns:
            ToolResult with raw HTML content
        """
        try:
            session = self._get_session()
            response = session.get(url, timeout=self.timeout)
            response.raise_for_status()

            html = response.text
            if len(html) > self.max_content_length:
                html = html[:self.max_content_length] + "\n<!-- truncated -->"

            return ToolResult(
                success=True,
                output=html,
                metadata={
                    "url": url,
                    "content_type": response.headers.get("Content-Type", ""),
                    "content_length": len(html),
                }
            )

        except requests.RequestException as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to fetch URL: {str(e)}"
            )

    def _extract_links(self, url: str, absolute: bool = True) -> ToolResult:
        """Extract all links from a page.

        Args:
            url: URL to fetch and extract links from
            absolute: If True, convert relative links to absolute

        Returns:
            ToolResult with list of links
        """
        if not HAS_BS4:
            return ToolResult(
                success=False,
                output=None,
                error="beautifulsoup4 required. Install with: pip install beautifulsoup4"
            )

        try:
            session = self._get_session()
            response = session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                text = a.get_text(strip=True)

                # Convert to absolute URL if needed
                if absolute and not href.startswith(("http://", "https://", "mailto:", "tel:")):
                    href = urljoin(url, href)

                # Skip anchors and javascript
                if href.startswith(("#", "javascript:")):
                    continue

                links.append({
                    "url": href,
                    "text": text[:100] if text else "",
                })

            return ToolResult(
                success=True,
                output=links,
                metadata={
                    "url": url,
                    "link_count": len(links),
                }
            )

        except requests.RequestException as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to fetch URL: {str(e)}"
            )

    def _extract_tables(self, url: str) -> ToolResult:
        """Extract tables from a web page.

        Args:
            url: URL to fetch and extract tables from

        Returns:
            ToolResult with list of tables (each as list of rows)
        """
        if not HAS_BS4:
            return ToolResult(
                success=False,
                output=None,
                error="beautifulsoup4 required. Install with: pip install beautifulsoup4"
            )

        try:
            session = self._get_session()
            response = session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            tables = []
            for table in soup.find_all("table"):
                table_data = []

                # Extract headers
                headers = []
                thead = table.find("thead")
                if thead:
                    for th in thead.find_all(["th", "td"]):
                        headers.append(th.get_text(strip=True))
                    if headers:
                        table_data.append(headers)

                # Extract rows
                tbody = table.find("tbody") or table
                for tr in tbody.find_all("tr"):
                    row = []
                    for cell in tr.find_all(["td", "th"]):
                        row.append(cell.get_text(strip=True))
                    if row and row != headers:  # Skip if same as header
                        table_data.append(row)

                if table_data:
                    tables.append(table_data)

            return ToolResult(
                success=True,
                output=tables,
                metadata={
                    "url": url,
                    "table_count": len(tables),
                }
            )

        except requests.RequestException as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Failed to fetch URL: {str(e)}"
            )
