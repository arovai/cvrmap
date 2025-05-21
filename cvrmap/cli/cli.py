#!/usr/bin/env python3
"""BIDS app to compute maps of CVR
"""

import warnings

from cvrmap.cvrmap import main
from cvrmap.utils.tools import setup_terminal_colors

setup_terminal_colors()
warnings.simplefilter("once")

if __name__ == "__main__":
    main()
