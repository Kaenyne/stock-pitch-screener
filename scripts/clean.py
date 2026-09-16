"""Reclaim regenerable churn. Safe by default; opt in to the aggressive sweeps.

    python scripts/clean.py                # __pycache__ + .pytest_cache (always safe)
    python scripts/clean.py --logs         # + old output/*.log (keeps newest)
    python scripts/clean.py --artifacts    # + output/tearsheets, output/details (regen next full run)
    python scripts/clean.py --all          # everything above
    python scripts/clean.py --all --dry-run  # show what would go, delete nothing

Never touches: data/ (edgar zips + parquet caches) or the CSV outputs.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
KEEP_LOG = None  # always keeps the newest log


def _size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def _rm(p: Path, dry: bool) -> int:
    n = _size(p)
    if not dry:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", action="store_true", help="clear old output/*.log (keeps newest)")
    ap.add_argument("--artifacts", action="store_true", help="clear output/tearsheets + output/details")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    logs, artifacts = a.logs or a.all, a.artifacts or a.all
    dry, freed = a.dry_run, 0
    tag = "[dry-run] would remove" if dry else "removed"

    # 1. byte-compiled caches — always, always regenerable
    for d in list(ROOT.rglob("__pycache__")) + list(ROOT.rglob(".pytest_cache")):
        if ".venv" in d.parts:
            continue
        freed += _rm(d, dry)
        print(f"{tag}: {d.relative_to(ROOT)}")

    # 2. old logs — keep the newest and the named current-run log
    if logs and OUT.is_dir():
        all_logs = sorted(OUT.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        keep = {all_logs[0].name if all_logs else None, KEEP_LOG}
        for lg in all_logs:
            if lg.name not in keep:
                freed += _rm(lg, dry)
                print(f"{tag}: {lg.relative_to(ROOT)}")

    # 3. per-company artifacts — regenerated every full run
    if artifacts:
        for sub in ("tearsheets", "details"):
            d = OUT / sub
            if d.is_dir():
                for f in d.iterdir():
                    freed += _rm(f, dry)
                print(f"{tag}: {d.relative_to(ROOT)}/* ")

    print(f"\n{'would free' if dry else 'freed'}: {freed / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
