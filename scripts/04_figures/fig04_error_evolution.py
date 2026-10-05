#!/usr/bin/env python3
"""Fig 4: parameter-precision time series (max absolute log estimated/true ratio)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
import numpy as np
import seaborn as sns

from _common import (
	DASHED_LINESTYLE,
	MAIN_TEXT_NUM_AGENT,
	MIN_EIGENVALUE_BOTTOM_MODELS,
	MIN_EIGENVALUE_TOP_MODELS,
	_load_truth_params,
	compute_max_abs_log_parameter_error,
	mean_std_series,
	PaperRecord,
	model_palette,
	plot_center_band,
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
	errors: list[float] = []
	for estimate in payload.get("estimates", []):
		error = compute_max_abs_log_parameter_error(
			parameter_mapping,
			estimate.get("best_params", {}),
			truth_params,
		)
		if error is None:
			continue
		time_seconds = float(estimate.get("time_seconds", float("nan")))
		# Zero error cannot enter the log-space (geometric) statistics.
		if np.isfinite(time_seconds) and np.isfinite(error) and error > 0.0:
			times.append(time_seconds)
			errors.append(error)

	if len(times) < 2:
		return None
	return np.asarray(times, dtype=float), np.asarray(errors, dtype=float)


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
		# Geometric mean and geometric-std band: statistics are taken on log(error).
		error_series = [
			series
			for record in cell
			if (series := _record_parameter_accuracy_series(record)) is not None
		]
		series = mean_std_series([(times, np.log(errors)) for times, errors in error_series])
		if series is None:
			continue
		times, log_mean, log_std = series
		finite_mask = np.isfinite(times) & np.isfinite(log_mean) & np.isfinite(log_std)
		times = times[finite_mask]
		log_mean = log_mean[finite_mask]
		log_std = log_std[finite_mask]
		if times.size < 2:
			continue

		present_groups.add(group)
		tick_times.append(times)
		plot_center_band(
			ax,
			times,
			np.exp(log_mean),
			np.exp(log_mean - log_std),
			np.exp(log_mean + log_std),
			palette.get(group, "#777777"),
		)

	if not present_groups:
		print(f"[{output_path.parent.parent.name}] No error time series to plot; skipping.")
		plt.close(fig)
		return

	ax.set_yscale(yscale)
	if yscale == "log":
		ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
		ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
		ax.yaxis.set_minor_formatter(NullFormatter())
	xticks = np.unique(np.round(np.concatenate(tick_times)))
	ax.set_xticks(xticks)
	ax.set_xlim(left=0.0)
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Worst-case relative error")
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
