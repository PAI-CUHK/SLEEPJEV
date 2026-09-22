"""Allow ``python -m sleepjev`` to use the public CLI."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
