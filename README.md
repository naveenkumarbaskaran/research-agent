# Research Agent AI

An AI-powered research agent that performs deep web research, verifies claims across multiple sources, and generates cited Markdown reports — all driven by [Claude claude-sonnet-4-6](https://docs.anthropic.com/en/docs/about-claude/models) via the Anthropic SDK.

## Features

- **Web search** using DuckDuckGo (no API key required)
- **Source fetching** via `httpx` with clean HTML-to-text conversion
- **Multi-step agentic loop** — Claude decides when to search, fetch, and write
- **Claim verification** across multiple sources at `deep` depth
- **Cited Markdown reports** with numbered inline citations and a References section
- **Rich CLI** with progress spinner and report preview

## Installation

```bash
pip install research-agent-ai
```

Or from source:

```bash
git clone https://github.com/example/research-agent-ai
cd research-agent-ai
pip install -e .
```

## Setup

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### CLI

```bash
# Deep research (default) — searches multiple sources, cross-verifies claims
research "What are the latest advances in quantum computing?"

# Shallow research — quick overview from 2-3 sources
research "History of the Roman Empire" --depth shallow

# Custom output file
research "Climate change solutions" --output climate_report.md

# Verbose mode — shows tool calls
research "Large language models" --verbose

# Use a different model
research "AI safety" --model claude-opus-4-8
```

### Python API

```python
from research_agent import ResearchAgent

agent = ResearchAgent(verbose=True)
report_path = agent.research(
    topic="Advances in renewable energy storage",
    depth="deep",
    output_path="energy_report.md",
)
print(f"Report saved to: {report_path}")
```

## CLI Options

| Option | Default | Description |
|---|---|---|
| `TOPIC` | *(required)* | Research question or subject |
| `--depth` | `deep` | `shallow` (2-3 sources) or `deep` (5-10 sources) |
| `--output` | `report.md` | Output file path |
| `--model` | `claude-sonnet-4-6` | Anthropic model ID |
| `--verbose` / `-v` | off | Show tool calls and intermediate steps |

## How It Works

The agent uses Claude claude-sonnet-4-6 in an agentic loop with three tools:

1. **`web_search(query)`** — Queries DuckDuckGo and returns titles, URLs, and snippets.
2. **`fetch_url(url)`** — Downloads a page and strips it to clean text via regex-based HTML parsing.
3. **`write_report(path, content)`** — Saves the final Markdown report to disk.

The loop runs until Claude calls `write_report`, producing a report with:
- Executive summary
- Structured sections with inline citations `[1]`, `[2]`, ...
- Source cross-verification notes (at `deep` depth)
- Numbered References section

## Architecture

```
research_agent/
├── __init__.py      # Package exports
├── agent.py         # ResearchAgent + tool implementations + agentic loop
├── fetcher.py       # WebFetcher: httpx + regex HTML-to-text
└── cli.py           # Click CLI with Rich output
```

## Development

```bash
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .
```

## License

MIT
