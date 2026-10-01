#!/usr/bin/env python3
from __future__ import annotations

import sys

from audit_baseline_call_flow import main


if __name__ == "__main__":
    if "--baseline" not in sys.argv:
        sys.argv.extend(["--baseline", "MIRIX"])
    raise SystemExit(main())
