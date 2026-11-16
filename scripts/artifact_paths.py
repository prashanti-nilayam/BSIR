"""Canonical artifact paths with clean-room override support.

The released bundle remains the default source and destination.  The
reproduction driver sets the two environment variables below to write into an
isolated output tree while continuing to read the immutable released traces.
"""
from __future__ import annotations

import os
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
TRACES = Path(os.environ.get("BORROWED_STRENGTH_TRACES", REPO / "traces")).resolve()
RESULTS = Path(os.environ.get("BORROWED_STRENGTH_RESULTS", REPO / "results")).resolve()

