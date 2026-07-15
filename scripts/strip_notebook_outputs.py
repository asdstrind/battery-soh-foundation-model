"""Remove execution state from notebooks or verify that notebooks are clean."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def clean_notebook(path: Path, check: bool) -> bool:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True
        metadata = cell.get("metadata", {})
        for key in ("ExecuteTime", "execution", "collapsed", "scrolled"):
            if key in metadata:
                metadata.pop(key)
                changed = True

    if changed and not check:
        path.write_text(
            json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    dirty = [path for path in root.rglob("*.ipynb") if clean_notebook(path, args.check)]
    if args.check and dirty:
        print("Notebooks contain outputs or execution state:")
        for path in dirty:
            print(f"- {path}")
        return 1
    print(f"Checked {len(list(root.rglob('*.ipynb')))} notebooks; changed {len(dirty)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

