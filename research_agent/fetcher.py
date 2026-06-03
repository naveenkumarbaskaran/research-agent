"""WebFetcher: httpx-based URL fetcher with HTML-to-text stripping."""

from __future__ import annotations

import re
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Default headers mimicking a real browser to avoid bot-blocks
# ---------------------------------------------------------------------------

DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
}

# ---------------------------------------------------------------------------
# Tags whose inner content should be dropped entirely
# ---------------------------------------------------------------------------

_DROP_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "iframe",
    "svg",
    "canvas",
    "figure",
)


class WebFetcher:
    """Fetches URLs via httpx and converts HTML to readable plain text.

    Only standard-library regex is used for HTML parsing — no external
    parser dependency (BeautifulSoup, lxml, etc.) is required.
    """

    def __init__(
        self,
        timeout: float = 20.0,
        max_redirects: int = 5,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.timeout = timeout
        self.headers = {**DEFAULT_HEADERS, **(headers or {})}
        self._client = httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            max_redirects=max_redirects,
            headers=self.headers,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, url: str) -> dict[str, Any]:
        """Fetch *url* and return a dict with keys: title, text, url, error.

        On success *error* is None. On failure *text* is empty and *error*
        contains a human-readable description.
        """
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            return {
                "url": url,
                "title": "",
                "text": "",
                "error": f"HTTP {exc.response.status_code}: {exc.response.reason_phrase}",
            }
        except httpx.RequestError as exc:
            return {
                "url": url,
                "title": "",
                "text": "",
                "error": f"Request error: {exc}",
            }

        content_type = response.headers.get("content-type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            # Return raw text for non-HTML content (JSON, XML, plain text)
            return {
                "url": str(response.url),
                "title": "",
                "text": response.text[:20_000],
                "error": None,
            }

        html = response.text
        title = self._extract_title(html)
        text = self._html_to_text(html)

        return {
            "url": str(response.url),
            "title": title,
            "text": text,
            "error": None,
        }

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()

    def __enter__(self) -> "WebFetcher":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # HTML parsing helpers (regex-based, no external dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_title(html: str) -> str:
        """Extract text from the <title> element."""
        match = re.search(r"<title[^>]*>([^<]*)</title>", html, re.IGNORECASE)
        if match:
            return _decode_entities(match.group(1).strip())
        return ""

    def _html_to_text(self, html: str) -> str:
        """Convert HTML to readable plain text.

        Steps:
        1. Drop noise tags (script, style, nav ...) with their contents.
        2. Replace block-level elements with newlines.
        3. Strip all remaining tags.
        4. Decode HTML entities.
        5. Collapse whitespace.
        """
        text = html

        # 1. Remove noise sections entirely (with their content)
        for tag in _DROP_TAGS:
            text = re.sub(
                rf"<{tag}[\s>][\s\S]*?</{tag}>",
                " ",
                text,
                flags=re.IGNORECASE,
            )

        # 2. Replace block-level tags with newlines
        block_tags = (
            "p", "div", "section", "article", "main",
            "h1", "h2", "h3", "h4", "h5", "h6",
            "li", "dt", "dd", "blockquote", "pre",
            "tr", "td", "th", "br",
        )
        for tag in block_tags:
            # Opening tags become newlines
            text = re.sub(rf"<{tag}[\s/>][^>]*>", "\n", text, flags=re.IGNORECASE)
            # Closing tags become newlines
            text = re.sub(rf"</{tag}>", "\n", text, flags=re.IGNORECASE)

        # 3. Strip all remaining HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # 4. Decode HTML entities
        text = _decode_entities(text)

        # 5. Normalise whitespace
        # Collapse runs of spaces/tabs to a single space
        text = re.sub(r"[^\S\n]+", " ", text)
        # Remove space at line boundaries
        text = re.sub(r" ?\n ?" , "\n", text)
        # Collapse more than two consecutive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


# ---------------------------------------------------------------------------
# HTML entity decoder (handles named + numeric entities)
# ---------------------------------------------------------------------------

_NAMED_ENTITIES: dict[str, str] = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
    "nbsp": " ",
    "mdash": "\u2014",
    "ndash": "\u2013",
    "lsquo": "\u2018",
    "rsquo": "\u2019",
    "ldquo": "\u201c",
    "rdquo": "\u201d",
    "hellip": "\u2026",
    "copy": "\u00a9",
    "reg": "\u00ae",
    "trade": "\u2122",
    "euro": "\u20ac",
    "pound": "\u00a3",
    "yen": "\u00a5",
    "cent": "\u00a2",
}


def _decode_entities(text: str) -> str:
    """Decode HTML character references (&amp;, &#160;, &#x0a;, etc.)."""
    # Numeric decimal: &#160;
    text = re.sub(
        r"&#(\d+);",
        lambda m: chr(int(m.group(1))),
        text,
    )
    # Numeric hex: &#x00a0;
    text = re.sub(
        r"&#x([0-9a-fA-F]+);",
        lambda m: chr(int(m.group(1), 16)),
        text,
    )
    # Named entities
    text = re.sub(
        r"&([a-zA-Z]+);",
        lambda m: _NAMED_ENTITIES.get(m.group(1).lower(), m.group(0)),
        text,
    )
    return text
