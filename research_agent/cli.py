"""Command-line interface for the research agent.

Usage examples:
    research "What are the latest advances in quantum computing?" --depth deep
    research "History of the Roman Empire" --depth shallow --output roman_empire.md
"""

from __future__ import annotations

import sys
import time

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.text import Text

from .agent import ResearchAgent

console = Console()


@click.command()
@click.argument("topic")
@click.option(
    "--depth",
    type=click.Choice(["shallow", "deep"], case_sensitive=False),
    default="deep",
    show_default=True,
    help="Research depth: 'shallow' for a quick overview, 'deep' for thorough analysis.",
)
@click.option(
    "--output",
    default="report.md",
    show_default=True,
    help="Output file path for the generated Markdown report.",
)
@click.option(
    "--model",
    default="claude-sonnet-4-6",
    show_default=True,
    help="Anthropic model to use.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Print tool calls and intermediate steps.",
)
def main(
    topic: str,
    depth: str,
    output: str,
    model: str,
    verbose: bool,
) -> None:
    """Run AI-powered deep research on TOPIC and save a cited Markdown report.

    TOPIC is the research question or subject to investigate.
    """
    console.print()
    console.print(
        Panel(
            Text(topic, style="bold cyan"),
            title="[bold]Research Agent[/bold]",
            subtitle=f"depth={depth}  model={model}",
            border_style="blue",
        )
    )
    console.print()

    agent = ResearchAgent(model=model, verbose=verbose)

    start = time.monotonic()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Researching", total=None)

        try:
            report_path = agent.research(topic, depth=depth, output_path=output)
        except anthropic_error() as exc:
            console.print(f"[red]Anthropic API error:[/red] {exc}")
            sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Research interrupted by user.[/yellow]")
            sys.exit(130)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Unexpected error:[/red] {exc}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        finally:
            progress.remove_task(task)

    elapsed = time.monotonic() - start

    console.print(
        Panel(
            f"Report saved to [bold green]{report_path}[/bold green]\n"
            f"Time elapsed: {elapsed:.1f}s",
            title="[bold green]Done[/bold green]",
            border_style="green",
        )
    )
    console.print()

    # Preview the first 60 lines of the report
    try:
        from pathlib import Path

        content = Path(report_path).read_text(encoding="utf-8")
        preview_lines = content.splitlines()[:60]
        preview = "\n".join(preview_lines)
        if len(content.splitlines()) > 60:
            preview += "\n\n*... (truncated — open the file to read the full report)*"
        console.print(Markdown(preview))
    except Exception:  # noqa: BLE001
        pass


def anthropic_error():
    """Return the anthropic.APIError class, or a generic Exception if not installed."""
    try:
        import anthropic
        return anthropic.APIError
    except ImportError:
        return Exception


if __name__ == "__main__":
    main()
