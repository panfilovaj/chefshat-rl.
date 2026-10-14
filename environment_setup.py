# AI-assisted setup helper. Does not modify the official environment.
import os
import sys
from pathlib import Path


def configure_source():
    """Use an explicit official source tree, or retain an existing PYTHONPATH."""
    configured = os.environ.get("CHEFSHAT_SRC")
    if configured:
        source = Path(configured).expanduser().resolve()
        if not all((source / name).is_dir() for name in ("agents", "rooms", "core")):
            raise RuntimeError("CHEFSHAT_SRC must point to the official repository's src folder.")
        sys.path.insert(0, str(source))
