"""Entry point kept for backwards compatibility (`python src/main.py ...`).

All real logic lives in the :mod:`news_agent` package; this shim only makes sure
``src/`` is importable when the file is executed directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from news_agent.cli import main as _cli_main  # noqa: E402


def main() -> int:
    return _cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
