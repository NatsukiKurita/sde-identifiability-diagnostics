#!/usr/bin/env python3
"""Spaghetti min-eigenvalue charts for TK2020 with four agents.

For N=2 the TK2020 cohesion and separation terms are collinear, so only
``k_coh - k_sep`` is identifiable. With N=4 the two terms separate and the
minimum eigenvalue grows, so these panels use the identifiable (slope=1) guide.
Outputs are written under ``results/04_supplemental_figures/tk2020_n4_spaghetti``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_figures"))

import matplotlib.pyplot as plt
import numpy as np

from _common import (
	SPAGHETTI_XSCALE,
	SPAGHETTI_YSCALE,
	PaperRecord,
	collect_paper_records,
	configure_plot_style,
	downsample_min_eigenvalue_series,
	model_palette,
	ordered_kinetics_from_set,
	prepare_min_eigenvalue_line,
	record_min_eigenvalue_series,
	save_paper_figure,
	style_min_eigenvalue_axis,
)
from src.paths import PROJECT_ROOT, RESULTS_ROOT

DEFAULT_SETTING = "setting_1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/04_supplemental_figures/tk2020_n4_spaghetti"
MODEL = "TK2020"
NUM_AGENTS = 4
FIGSIZE = (6.4, 4.0)
LINE_ALPHA = 0.2
LINEWIDTH = 0.3


def plot_spaghetti(records: list[PaperRecord], kinetics: str, output_path: Path) -> None:
	color = model_palette([MODEL])[MODEL]
	fig, ax = plt.subplots(figsize=FIGSIZE)
	x_values: list[np.ndarray] = []
	y_values: list[np.ndarray] = []
	for record in sorted(records, key=lambda item: item.base.sample_name):
		times, values = downsample_min_eigenvalue_series(*record_min_eigenvalue_series(record))
		times, values = prepare_min_eigenvalue_line(
			times,
			values,
			xscale=SPAGHETTI_XSCALE,
			yscale=SPAGHETTI_YSCALE,
		)
		if times.size < 2 or np.count_nonzero(np.isfinite(values)) < 2:
			continue
		x_values.append(times)
		y_values.append(values[np.isfinite(values)])
		ax.plot(times, values, color=color, linestyle="-", linewidth=LINEWIDTH, alpha=LINE_ALPHA)

	if not x_values:
		print(f"No spaghetti lines for {MODEL} {kinetics} N={NUM_AGENTS}; skipping.")
		plt.close(fig)
		return

	style_min_eigenvalue_axis(
		ax,
		xscale=SPAGHETTI_XSCALE,
		yscale=SPAGHETTI_YSCALE,
		x_values=x_values,
		y_values=y_values,
		identifiable=True,
	)
	ax.set_title(f"{MODEL} ({kinetics}, N={NUM_AGENTS})")
	fig.tight_layout()
	save_paper_figure(fig, output_path)


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
	records = [
		record
		for record in collect_paper_records(setting_dir)
		if record.group == MODEL and record.panel == NUM_AGENTS
	]
	if not records:
		print(f"[{args.setting}] No {MODEL} N={NUM_AGENTS} samples; skipping.")
		return
	print(f"Loaded {len(records)} {MODEL} N={NUM_AGENTS} samples from {args.setting}")

	output_dir = args.output_dir / args.setting
	for kinetics in ordered_kinetics_from_set({record.kinetics for record in records}):
		plot_spaghetti(
			[record for record in records if record.kinetics == kinetics],
			kinetics,
			output_dir / f"spaghetti_{MODEL}_N{NUM_AGENTS}_{kinetics}.pdf",
		)


if __name__ == "__main__":
	main()
