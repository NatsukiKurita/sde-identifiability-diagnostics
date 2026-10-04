#!/usr/bin/env python3
"""Fig 4: parameter-precision time series (max estimated/true ratio)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from _common import (
	DASHED_LINESTYLE,
	MAIN_TEXT_NUM_AGENT,
	MIN_EIGENVALUE_BOTTOM_MODELS,
	MIN_EIGENVALUE_LOG_FLOOR,
	MIN_EIGENVALUE_TOP_MODELS,
	_compute_parameter_ratio_bounds,
	_load_truth_params,
	mean_std_series,
	PaperRecord,
	model_palette,
	plot_mean_std_band,
	read_json,
	run_for_settings,
	save_paper_figure,
)

# Adjust this figure size independently.
FIGSIZE = (6.4, 4.0)
ERROR_PLOT_KINETICS_DIRS = (
	("Inertial", "Inertial"),
	("Overdamped", "Overdamped"),
)


def _record_parameter_accuracy_series(record: PaperRecord) -> tuple[np.ndarray, np.ndarray] | None:
	if record.result_dir is None:
		return None
	estimates_path = record.result_dir / "parameter_evolution.json"
	if not estimates_path.is_file():
		return None

	payload = read_json(estimates_path)
	parameter_mapping = payload.get("parameter_mapping", [])
	truth_params = _load_truth_params(record.base.setting, record.base.model_id)
	if truth_params is None:
		return None

	times: list[float] = []
	max_ratios: list[float] = []
	for estimate in payload.get("estimates", []):
		ratio_bounds = _compute_parameter_ratio_bounds(
			parameter_mapping,
			estimate.get("best_params", {}),
			truth_params,
		)
		if ratio_bounds is None:
			continue
		time_seconds = float(estimate.get("time_seconds", float("nan")))
		max_ratio = float(ratio_bounds[1])
		if np.isfinite(time_seconds) and np.isfinite(max_ratio):
			times.append(time_seconds)
			max_ratios.append(max_ratio)

	if len(times) < 2:
		return None
	return np.asarray(times, dtype=float), np.asarray(max_ratios, dtype=float)


def plot_precision_panel(
	filtered: list[PaperRecord],
	group_order: list[str],
	output_path: Path,
	*,
	yscale: str,
) -> None:
	palette = model_palette(group_order)
	fig, ax = plt.subplots(figsize=FIGSIZE)
	present_groups: set[str] = set()
	tick_times: list[np.ndarray] = []

	for group in group_order:
		cell = [record for record in filtered if record.group == group]
		series = mean_std_series(
			[
				error_series
				for record in cell
				if (error_series := _record_parameter_accuracy_series(record)) is not None
			]
		)
		if series is None:
			continue
		times, mean_values, std_values = series
		finite_mask = (
			np.isfinite(times)
			& np.isfinite(mean_values)
			& np.isfinite(std_values)
			& (mean_values > 0.0)
		)
		times = times[finite_mask]
		mean_values = mean_values[finite_mask]
		std_values = std_values[finite_mask]
		if times.size < 2:
			continue

		present_groups.add(group)
		tick_times.append(times)
		plot_mean_std_band(
			ax,
			times,
			mean_values,
			std_values,
			palette.get(group, "#777777"),
			clip_lower=MIN_EIGENVALUE_LOG_FLOOR if yscale == "log" else 0.0,
		)

	if not present_groups:
		print(f"[{output_path.parent.parent.name}] No error time series to plot; skipping.")
		plt.close(fig)
		return

	ax.set_yscale(yscale)
	ax.axhline(1.0, color="0.55", linestyle=DASHED_LINESTYLE, linewidth=1.0, zorder=0)
	xticks = np.unique(np.round(np.concatenate(tick_times)))
	ax.set_xticks(xticks)
	ax.set_xlim(left=0.0)
	ax.set_xlabel("Time (s)")
	ax.set_ylabel(r"$\max(p_{\mathrm{opt}}/p_{\mathrm{true}})$")
	ax.grid(True, which="both" if yscale == "log" else "major", linestyle=DASHED_LINESTYLE, alpha=0.25)
	sns.despine(ax=ax)
	fig.tight_layout()
	save_paper_figure(fig, output_path)


def plot_for_setting(setting_dir: Path, records: list[PaperRecord], _args) -> None:
	for kinetics, kinetics_dir in ERROR_PLOT_KINETICS_DIRS:
		filtered = [
			record
			for record in records
			if record.panel == MAIN_TEXT_NUM_AGENT and record.kinetics == kinetics
		]
		if not filtered:
			print(
				f"[{setting_dir.name}] No N={MAIN_TEXT_NUM_AGENT} {kinetics} data for precision plot; skipping."
			)
			continue

		output_dir = setting_dir / "figures" / "error_plot" / kinetics_dir
		plot_precision_panel(
			filtered,
			MIN_EIGENVALUE_TOP_MODELS,
			output_dir / "error_plot_top.pdf",
			yscale="linear",
		)
		plot_precision_panel(
			filtered,
			MIN_EIGENVALUE_BOTTOM_MODELS,
			output_dir / "error_plot_bottom.pdf",
			yscale="log",
		)


if __name__ == "__main__":
	run_for_settings(
		plot_for_setting,
		description="Parameter-precision time series (max estimated/true ratio).",
	)
