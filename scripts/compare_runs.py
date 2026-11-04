"""CLI entry point to compare Baseline, Improved (Blind), and Oracle-Mask runs."""
import argparse
import os
from pathlib import Path
import sys

os.environ["MPLBACKEND"] = "Agg"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.comparison import compare_profiles


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/train_smoke/manifest.json")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--backup-dir", type=Path, default=None, help="Restore backups only from this explicit directory; omitted = local runs only")
    parser.add_argument("--baseline", type=str, default=None, help="Baseline profile hash or path")
    parser.add_argument("--improved", type=str, default=None, help="Improved or Oracle-Mask profile hash or path")
    parser.add_argument("--oracle", type=str, default=None, help="Optional Oracle-Mask profile hash or path")
    parser.add_argument("--output-grid", type=Path, default=ROOT / "outputs/runs/comparison_grid.jpg")
    args = parser.parse_args()

    compare_profiles(
        manifest_path=args.manifest,
        runs_root=args.runs_root,
        output_grid=args.output_grid,
        backup_dir=args.backup_dir,
        baseline_spec=args.baseline,
        improved_spec=args.improved,
        oracle_spec=args.oracle,
    )


if __name__ == "__main__":
    main()
