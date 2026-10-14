# AI-assisted provenance recorder. Records the source actually selected.
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main():
    configured = os.environ.get("CHEFSHAT_SRC")
    if not configured:
        raise RuntimeError("Set CHEFSHAT_SRC to the official src directory first.")
    source = Path(configured).resolve()
    if not all((source / name).is_dir() for name in ("agents", "rooms", "core")):
        raise RuntimeError("Invalid official src directory.")
    hashes = {
        str(p.relative_to(source)).replace(os.sep, "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(source.rglob("*.py"))
    }
    commit = None
    try:
        if (source.parent / ".git").exists():
            commit = subprocess.check_output(
                ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "environment_git_commit": commit,
        "source_fingerprint": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "source_python_sha256": hashes,
        "packages": {name: importlib.metadata.version(name) for name in
                     ["chefshatgym", "torch", "numpy", "pandas", "matplotlib", "websockets"]},
        "note": "A ZIP has no git commit. Record its originating commit separately before submission."
    }
    Path("results").mkdir(exist_ok=True)
    target = Path("results") / "environment_manifest.json"
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
