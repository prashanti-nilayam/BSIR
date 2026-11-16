#!/usr/bin/env python3
"""Create or verify SHA-256 hashes for all released trace artifacts."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from artifact_paths import REPO, TRACES

DEFAULT = REPO / "MANIFEST.sha256"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def entries():
    return [(Path("traces") / path.relative_to(TRACES), digest(path))
            for path in sorted(TRACES.rglob("*")) if path.is_file()]


def write(path):
    rows = entries()
    path.write_text("".join(f"{value}  {relative.as_posix()}\n"
                            for relative, value in rows))
    print(f"wrote {path} ({len(rows)} traces)")


def verify(path):
    expected = {}
    for line in path.read_text().splitlines():
        value, relative = line.split("  ", 1)
        expected[relative] = value
    actual = {relative.as_posix(): value for relative, value in entries()}
    problems = []
    for relative in sorted(set(expected) | set(actual)):
        if expected.get(relative) != actual.get(relative):
            problems.append(relative)
    if problems:
        print(f"FAIL: {len(problems)} trace manifest mismatch(es)")
        for relative in problems:
            print("  ", relative)
        return 1
    print(f"PASS: {len(actual)} trace hashes match {path.name}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.write:
        write(args.manifest.resolve())
        return 0
    return verify(args.manifest.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
