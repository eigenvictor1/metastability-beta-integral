"""`python3 -m higgsflow ...`: same as the `higgsflow` command (see cli.py)."""
import sys

from .cli import main

sys.exit(main())
