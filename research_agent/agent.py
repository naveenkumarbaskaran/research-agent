"""ResearchAgent: multi-step research loop using Claude claude-sonnet-4-6 with tool use.

Flow:
  1. web_search  - discover sources via DuckDuckGo
  2. fetch_url   - retrieve and strip each source page
  3. (loop)      - Claude decides when enough evidence is gathered
  4. write_report - synthesise a cited Markdown report
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import anthropic

from .fetcher import WebFetcher

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": (
            "Search the web for information about a topic using DuckDuckGo. "
            "Returns a list of result titles, URLs, and snippets. "
            "Call this when you need to discover sources or find current information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_url",
        "description": (
            "Fetch the content of a URL and return it as plain text. "
            "Use this to read the full content of a source page discovered via web_search. "
            "Returns the page title and cleaned text body."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL (https://...) to fetch.",
                }
            },
            "required": ["url"],
        },
    },
    {
        "name": "write_report",
        "description": (
            "Write the final research report to a file. "
            "Call this exactly once when you have gathered sufficient evidence and are ready "
            "to produce the final cited report. The content must be well-structured Markdown "
            "with inline citations like [1], [2], etc., and a References section at the end."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path where the report should be saved, e.g. 'report.md'.",
                },
                "content": {
                    "type": "string",
                    "description": "Full Markdown content of the report including citations.",
                },
            },
            "required": ["path", "content"],
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert research analyst. Your job is to:

1. Search for information on the given topic using web_search.
2. Fetch and read the most relevant source pages using fetch_url.
3. Cross-verify key claims across multiple sources.
4. Synthesise a well-structured, cited Markdown report using write_report.

Guidelines:
- For a 'shallow' depth: perform 1-2 searches and fetch 2-3 sources.
- For a 'deep' depth: perform 3-5 searches and fetch 5-10 sources, cross-checking claims.
- Always include a References section with numbered citations.
- When claims conflict across sources, note the discrepancy in the report.
- Write the report in a professional, neutral tone.
- Call write_report exactly once when you are satisfied with the evidence gathered.
"""

# ---------------------------------------------------------------------------
# ResearchAgent
# ---------------------------------------------------------------------------


class ResearchAgent:
    """Agentic research loop backed by Claude claude-sonnet-4-6."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.client = anthropic.Anthropic()
        self.fetcher = WebFetcher()
        self._report_written: bool = False
        self._report_path: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research(self, topic: str, depth: str = "deep", output_path: str = "report.md") -> str:
        """Run the full research loop and return the path of the written report.

        Args:
            topic: The research question or topic.
            depth: 'shallow' for a quick overview, 'deep' for thorough research.
            output_path: Where to save the final Markdown report.

        Returns:
            The file path where the report was saved.
        """
        self._report_written = False
        self._report_path = None

        user_message = (
            f"Research the following topic and produce a cited Markdown report.\n\n"
            f"Topic: {topic}\n"
            f"Depth: {depth}\n"
            f"Output file: {output_path}\n\n"
            "Begin by searching for relevant information, then fetch and read the most "
            "important sources. When you have gathered enough evidence, write the final "
            "report using write_report."
        )

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_message}
        ]

        if self.verbose:
            print(f"[agent] Starting research: {topic!r} (depth={depth})")

        # Agentic loop
        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=TOOLS,  # type: ignore[arg-type]
                messages=messages,
            )

            if self.verbose:
                print(f"[agent] stop_reason={response.stop_reason}")

            # Append assistant turn
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                break

            if response.stop_reason != "tool_use":
                # Unexpected stop; bail
                break

            # Execute tool calls
            tool_results = self._execute_tools(response.content)
            messages.append({"role": "user", "content": tool_results})

            # Stop after write_report is called
            if self._report_written:
                # One more LLM turn so Claude can acknowledge, then stop
                # (the loop condition will catch end_turn on the next iteration)
                pass

        if self._report_path is None:
            # Claude finished without calling write_report; extract text and save
            text = self._extract_text(response.content)
            path = Path(output_path)
            path.write_text(text, encoding="utf-8")
            self._report_path = str(path)
            if self.verbose:
                print(f"[agent] Saved fallback report to {self._report_path}")

        return self._report_path  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _execute_tools(
        self, content: list[Any]
    ) -> list[dict[str, Any]]:
        """Execute all tool_use blocks and return tool_result list."""
        results: list[dict[str, Any]] = []
        for block in content:
            if block.type != "tool_use":
                continue
            tool_name: str = block.name
            tool_input: dict[str, Any] = block.input
            tool_use_id: str = block.id

            if self.verbose:
                print(f"[tool] {tool_name}({json.dumps(tool_input, ensure_ascii=False)[:120]})")

            try:
                output = self._dispatch(tool_name, tool_input)
                is_error = False
            except Exception as exc:  # noqa: BLE001
                output = f"Error executing {tool_name}: {exc}"
                is_error = True
                if self.verbose:
                    print(f"[tool] ERROR: {exc}")

            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": str(output),
                    "is_error": is_error,
                }
            )
        return results

    def _dispatch(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        if tool_name == "web_search":
            return self._tool_web_search(tool_input["query"])
        if tool_name == "fetch_url":
            return self._tool_fetch_url(tool_input["url"])
        if tool_name == "write_report":
            return self._tool_write_report(tool_input["path"], tool_input["content"])
        raise ValueError(f"Unknown tool: {tool_name}")

    # ------------------------------------------------------------------
    # Individual tools
    # ------------------------------------------------------------------

    def _tool_web_search(self, query: str) -> str:
        """DuckDuckGo instant-answer / HTML search."""
        import urllib.parse
        import urllib.request

        encoded = urllib.parse.quote_plus(query)
        # DuckDuckGo HTML endpoint (no API key required)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (research-agent/0.1)",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")
        except Exception as exc:
            return f"Search failed: {exc}"

        return self._parse_ddg_results(html, max_results=8)

    @staticmethod
    def _parse_ddg_results(html: str, max_results: int = 8) -> str:
        """Extract result titles, URLs, and snippets from DuckDuckGo HTML."""
        # Extract result blocks
        result_blocks = re.findall(
            r'<div class="result[^"]*".*?</div>\s*</div>\s*</div>',
            html,
            re.DOTALL,
        )

        results: list[dict[str, str]] = []
        for block in result_blocks[:max_results]:
            # Title + URL
            title_match = re.search(
                r'class="result__title".*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                block,
                re.DOTALL,
            )
            snippet_match = re.search(
                r'class="result__snippet"[^>]*>(.*?)</span>',
                block,
                re.DOTALL,
            )

            if not title_match:
                continue

            raw_url = title_match.group(1)
            title = re.sub(r"<[^>]+>", "", title_match.group(2)).strip()
            snippet = re.sub(r"<[^>]+>", "", snippet_match.group(1)).strip() if snippet_match else ""

            # DuckDuckGo sometimes returns redirect URLs; decode them
            if raw_url.startswith("//duckduckgo.com/l/"):
                import urllib.parse
                qs = urllib.parse.parse_qs(urllib.parse.urlparse("https:" + raw_url).query)
                raw_url = qs.get("uddg", [raw_url])[0]

            results.append({"title": title, "url": raw_url, "snippet": snippet})

        if not results:
            # Fallback: grab any <a> with http in href
            links = re.findall(r'href="(https?://[^"]+)"', html)
            unique: list[str] = []
            for lnk in links:
                if lnk not in unique and "duckduckgo" not in lnk:
                    unique.append(lnk)
                    if len(unique) >= max_results:
                        break
            if unique:
                return "Search results (URLs only):\n" + "\n".join(unique)
            return "No search results found."

        lines = [f"Found {len(results)} result(s):\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r['title']}")
            lines.append(f"    URL: {r['url']}")
            if r["snippet"]:
                lines.append(f"    Snippet: {r['snippet']}")
            lines.append("")
        return "\n".join(lines)

    def _tool_fetch_url(self, url: str) -> str:
        """Fetch a URL and return cleaned text."""
        result = self.fetcher.fetch(url)
        if result["error"]:
            return f"Failed to fetch {url}: {result['error']}"
        title = result.get("title", "")
        text = result.get("text", "")
        # Truncate to avoid overwhelming context
        max_chars = 12_000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... content truncated ...]"
        header = f"Title: {title}\nURL: {url}\n\n" if title else f"URL: {url}\n\n"
        return header + text

    def _tool_write_report(self, path: str, content: str) -> str:
        """Write the report to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        self._report_written = True
        self._report_path = str(p)
        if self.verbose:
            print(f"[tool] Report written to {p} ({len(content)} chars)")
        return f"Report successfully written to {path} ({len(content)} characters)."

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(content: list[Any]) -> str:
        """Extract plain text from a content block list."""
        parts: list[str] = []
        for block in content:
            if hasattr(block, "type") and block.type == "text":
                parts.append(block.text)
        return "\n\n".join(parts)
