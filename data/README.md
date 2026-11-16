# Trace archive

The raw trace files are intentionally excluded from Git history. Obtain
`bsir-v1.0.0-traces.tar.gz` from the v1.0.0 GitHub release and run:

```bash
python scripts/fetch_traces.py --archive /path/to/bsir-v1.0.0-traces.tar.gz
```

The installer checks `trace_archive.sha256`, validates every archive member and
its manifest hash, and installs `traces/`. It refuses to overwrite existing data.
To verify an existing installation, run `python scripts/trace_manifest.py`.
The full reproduction ZIP already includes `traces/`.
