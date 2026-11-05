#!/usr/bin/env python3
"""pixelmap.py - camera pixel mapping CLI. The code lives in twinklybridge/mapping.py; run
`python3 -m twinklybridge.mapping --help` or this file from a checkout."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from twinklybridge.mapping import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
