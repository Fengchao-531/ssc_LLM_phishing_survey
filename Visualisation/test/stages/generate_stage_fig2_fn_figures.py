#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

VIS_ROOT = Path(__file__).resolve().parents[2]
if str(VIS_ROOT) not in sys.path:
    sys.path.insert(0, str(VIS_ROOT))

from generate_stage_fig2_fn_figures import main


if __name__ == "__main__":
    main()
