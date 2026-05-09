"""python -m ml  →  unified CLI."""

import sys

from ml.cli.main import main

if __name__ == "__main__":
    main(sys.argv[1:])
