"""Build the Scenario Inspector page from one or more ``inspect.json`` exports.

``python -m guillotine viewer out_a/inspect.json out_b/inspect.json --out inspector.html``
"""

from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("viewer_template.html")


def build(inspect_files: list[str | Path], out: str | Path) -> Path:
    data = [json.loads(Path(f).read_text()) for f in inspect_files]
    html = TEMPLATE.read_text().replace("/*DATA*/", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    out = Path(out)
    out.write_text(html)
    return out
