#!/usr/bin/env python3
"""Spaghetti min-eigenvalue charts for every social model and kinetics with N=4.

Layout and styling match ``scripts/04_figures/fig02_spaghetti.py`` (N=2 main text).
TK2020 at N=4 uses the identifiable slope=1 guide because cohesion and separation
decouple. Outputs under ``results/04_supplemental_figures/n4_spaghetti/<setting>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_figures"))

import fig02_spaghetti
from _common import collect_paper_records, configure_plot_style
from src.paths import PROJECT_ROOT, RESULTS_ROOT

DEFAULT_SETTING = "setting_1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/04_supplemental_figures/n4_spaghetti"
NUM_AGENTS = 4


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
	parser.add_argument("--setting", type=str, default=DEFAULT_SETTING)
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	args = parser.parse_args()

	setting_dir = args.results_root / args.setting
	if not setting_dir.is_dir():
		raise FileNotFoundError(f"Setting directory not found: {setting_dir}")

	configure_plot_style()
	records = collect_paper_records(setting_dir)
	if not records:
		print(f"No completed records found under {setting_dir}; skipping.")
		return
	print(f"Loaded {len(records)} samples from {args.setting}")
	fig02_spaghetti.plot_for_setting(
		setting_dir,
		records,
		args,
		num_agents=NUM_AGENTS,
		figures_dir=args.output_dir / args.setting,
	)


if __name__ == "__main__":
	main()
