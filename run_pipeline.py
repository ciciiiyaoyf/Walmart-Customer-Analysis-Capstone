"""Run the pipeline, an explicit range, or --from-clean for the committed snapshot."""
import argparse
import asyncio
import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", nargs="?", type=int, default=None)
    parser.add_argument("last", nargs="?", type=int, default=10)
    parser.add_argument("--from-clean", action="store_true", help="Start at 02 using committed clean tables; raw CSVs are not revalidated")
    args = parser.parse_args()
    first = args.first if args.first is not None else (2 if args.from_clean else 1)
    if args.from_clean and args.first is not None:
        parser.error("Use --from-clean or an explicit range, not both")
    if not 1 <= first <= args.last <= 10:
        parser.error("Expected 1 <= first <= last <= 10")
    root = Path(__file__).resolve().parent
    if first == 1 and not (root / "data" / "raw" / "customers.csv").exists():
        parser.error("Raw CSVs are not committed. Supply the 11 source CSVs or use --from-clean for the committed snapshot.")
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    notebooks = root / "notebooks"
    for path in sorted(p for p in notebooks.glob("[0-9][0-9]_*.ipynb") if first <= int(p.name[:2]) <= args.last):
        t0 = time.time()
        print(f"running {path.name}", flush=True)
        nb = nbformat.read(path, as_version=4)
        try:
            NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(notebooks)}}).execute()
        except Exception as exc:
            print(f"FAILED {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1
        nbformat.write(nb, path)
        print(f"ok      {path.name} ({time.time()-t0:.0f}s)", flush=True)
    print("pipeline complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
