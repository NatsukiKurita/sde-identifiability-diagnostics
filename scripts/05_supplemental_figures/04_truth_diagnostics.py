#!/usr/bin/env python3
"""Main-text diagnostic figures (fig01 sample paths, fig02 spaghetti) evaluated at the true parameters.

Identical in layout to ``scripts/04_figures/fig01_min_eigenvalue_timeseries.py`` and
``scripts/04_figures/fig02_spaghetti.py``, but the minimum eigenvalues are read from
``results/truth`` instead of ``results/optimum`` in ``result_time_series.h5``.
KR2021 is not split by estimated-m regime (at truth all samples are linear). TK2020
spaghetti omits the terminal linear inset.
Outputs are written under ``results/04_supplemental_figures/truth_diagnostics/<setting>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_figures"))

from _common import add_common_arguments, collect_paper_records, configure_plot_style, discover_setting_dirs
import fig01_min_eigenvalue_timeseries
import fig02_spaghetti
from src.paths import PROJECT_ROOT

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/04_supplemental_figures/truth_diagnostics"
EIGENVALUE_LABEL = "truth"
def main() -> None:
	parser = add_common_arguments(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
	parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
	args = parser.parse_args()

	configure_plot_style()
	setting_dirs = discover_setting_dirs(args.results_root, args.setting)
	if not setting_dirs:
		raise FileNotFoundError(f"No setting directories found under {args.results_root}")
	for setting_dir in setting_dirs:
		records = collect_paper_records(setting_dir, eigenvalue_label=EIGENVALUE_LABEL)
		if not records:
			print(f"No completed records found under {setting_dir}; skipping.")
			continue
		print(f"Loaded {len(records)} samples ({EIGENVALUE_LABEL}) from {setting_dir.name}")
		fig01_min_eigenvalue_timeseries.plot_for_setting(
			setting_dir,
			records,
			args,
			figures_dir=args.output_dir / setting_dir.name,
			kr2021_split_by_estimate=False,
		)
		fig02_spaghetti.plot_for_setting(
			setting_dir,
			records,
			args,
			figures_dir=args.output_dir / setting_dir.name,
			kr2021_split_by_estimate=False,
			tk2020_linear_tail_inset=False,
		)


if __name__ == "__main__":
	main()
