#!/usr/bin/env python3
"""Install the versioned trace archive after checking its SHA-256 and members."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
import urllib.request

REPO = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def install(archive):
    expected_hash = (REPO / "data/trace_archive.sha256").read_text().split()[0]
    if digest(archive) != expected_hash:
        raise ValueError("trace archive SHA-256 mismatch; nothing was extracted")
    expected = {}
    for line in (REPO / "MANIFEST.sha256").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        expected[name] = checksum
    destination = REPO / "traces"
    if destination.exists():
        raise FileExistsError("traces/ already exists; verify it with scripts/trace_manifest.py")
    with tempfile.TemporaryDirectory(prefix=".trace-install-", dir=REPO) as tmp:
        stage = Path(tmp)
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            names = [m.name for m in members if m.isfile()]
            if len(names) != len(set(names)) or set(names) != set(expected):
                raise ValueError("archive members differ from the trace manifest")
            for member in members:
                rel = Path(member.name)
                if rel.is_absolute() or ".." in rel.parts or not (member.isfile() or member.isdir()):
                    raise ValueError("unsupported archive member")
                if member.isdir():
                    if rel.parts[0] != "traces":
                        raise ValueError("unexpected directory in trace archive")
                    continue
                target = stage / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError("unreadable trace member")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                if digest(target) != expected[member.name]:
                    raise ValueError("trace member checksum mismatch: " + member.name)
        (stage / "traces").rename(destination)
    print(f"Installed and verified {len(expected)} trace artifacts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--archive", type=Path, help="downloaded trace archive")
    group.add_argument("--url", help="HTTPS URL of the versioned release asset")
    args = parser.parse_args()
    if args.archive:
        install(args.archive.resolve())
    else:
        if not args.url.startswith("https://"):
            parser.error("--url must use HTTPS")
        with tempfile.TemporaryDirectory(prefix="bsir-download-") as tmp:
            archive = Path(tmp) / "traces.tar.gz"
            urllib.request.urlretrieve(args.url, archive)
            install(archive)


if __name__ == "__main__":
    main()
