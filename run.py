"""Answer a question about a CSV dataset.

    python run.py --question "What is the mean revenue by region?" --verbose
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_analyst_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
