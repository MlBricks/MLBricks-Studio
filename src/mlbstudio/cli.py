"""Command-line launcher for MLB Studio."""

from __future__ import annotations

from . import Builder


def main() -> None:
    """Launch MLB Studio as the local application."""
    Builder().app()


if __name__ == "__main__":
    main()
