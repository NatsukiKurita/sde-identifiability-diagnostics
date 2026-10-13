#!/usr/bin/env python3
"""Fig 1: one-sample (or mean) min-eigenvalue series, identifiable vs unidentifiable."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from _common import (
	DASHED_LINESTYLE,
	KINETICS_PLOT_LINESTYLES,
	MAIN_TEXT_NUM_AGENT,
	MIN_EIGENVALUE_BOTTOM_MODELS,
	MIN_EIGENVALUE_LOG_FLOOR,
	MIN_EIGENVALUE_PLOT_SECONDS,
	MIN_EIGENVALUE_PLOT_START_SECONDS,
	MIN_EIGENVALUE_TOP_MODELS,
	SPECIAL_MIN_EIGENVALUE_SERIES,
	downsample_min_eigenvalue_series,
	mean_series,
	PaperRecord,
	add_slope_one_guide,
	add_split_line_legends,
	linestyle_for_linewidth,
	model_palette,
	ordered_kinetics_from_set,
	record_min_eigenvalue_series,
	run_for_settings,
	save_paper_figure,
	split_kr2021_regimes,
)

# Adjust this figure size independently.
FIGSIZE = (6.4, 4.0)
# Early transients reach about -47 and would hide the late-time behaviour.
UNIDENTIFIABLE_YLIM = (-3.0, 7.0)


def _min_eigenvalue_plot_series(
	records: list[PaperRecord],
	group: str,
	kinetics: str,
) -> tuple[np.ndarray, np.ndarray] | None:
	special_sample_names = SPECIAL_MIN_EIGENVALUE_SERIES.get((group, kinetics), ())
	for record in records:
		if record.base.sample_name in special_sample_names:
			series = record_min_eigenvalue_series(record)
			break
	else:
		series = mean_series([record_min_eigenvalue_series(record) for record in records])

	if series is None:
		return None
	return downsample_min_eigenvalue_series(*series)


def plot_min_eigenvalue_timeseries_panel(
	filtered: list[PaperRecord],
	group_order: list[str],
	output_path: Path,
	*,
	xscale: str,
	yscale: str,
	ylim: tuple[float, float] | None = None,
) -> None:
	kinetics_order = ordered_kinetics_from_set({record.kinetics for record in filtered})
	palette = model_palette(group_order)

	fig, ax = plt.subplots(figsize=FIGSIZE)
	present_groups: set[str] = set()
	present_kinetics: set[str] = set()
	y_values: list[np.ndarray] = []
	recoveries: list[tuple[float, float, tuple[float, float, float]]] = []
	for group in group_order:
		for kinetics in kinetics_order:
			cell = [
				record
				for record in filtered
				if record.group == group and record.kinetics == kinetics
			]
			series = _min_eigenvalue_plot_series(cell, group, kinetics)
			if series is None:
				continue
			times, values = series
			finite_mask = (
				np.isfinite(times)
				& np.isfinite(values)
				& (times >= MIN_EIGENVALUE_PLOT_START_SECONDS)
				& (times <= MIN_EIGENVALUE_PLOT_SECONDS)
			)
			times = times[finite_mask]
			values = values[finite_mask]
			if yscale == "log":
				positive_mask = values > MIN_EIGENVALUE_LOG_FLOOR
				recovery_indices = np.flatnonzero(positive_mask[1:] & ~positive_mask[:-1]) + 1
				recoveries.extend(
					(
						float(times[index]),
						float(values[index]),
						palette.get(group, (0.47, 0.47, 0.47)),
					)
					for index in recovery_indices
				)
				values = np.where(values > MIN_EIGENVALUE_LOG_FLOOR, values, np.nan)
			if times.size < 2 or np.count_nonzero(np.isfinite(values)) < 2:
				continue

			present_groups.add(group)
			present_kinetics.add(kinetics)
			y_values.append(values[np.isfinite(values)])
			linewidth = 1.15
			linestyle = KINETICS_PLOT_LINESTYLES.get(kinetics, ":")
			alpha = 0.9
			if group == "TK2020" and kinetics == "Inertial":
				linewidth = 1.15
				alpha = 0.75
			ax.plot(
				times,
				values,
				color=palette.get(group, "#777777"),
				linestyle=linestyle_for_linewidth(linestyle, linewidth),
				linewidth=linewidth,
				alpha=alpha,
			)

	if not present_groups:
		print(f"[{output_path.parent.parent.name}] No minimum-eigenvalue lines to plot; skipping.")
		plt.close(fig)
		return

	ax.set_xscale(xscale)
	ax.set_yscale(yscale)
	ax.set_xlim(left=MIN_EIGENVALUE_PLOT_START_SECONDS, right=MIN_EIGENVALUE_PLOT_SECONDS)
	if yscale == "log" and y_values:
		bottom = max(MIN_EIGENVALUE_LOG_FLOOR, min(float(np.nanmin(values)) for values in y_values))
		ax.set_ylim(bottom=bottom)
		recovery_bottom = ax.get_ylim()[0]
		for recovery_time, recovery_value, color in recoveries:
			ax.vlines(
				recovery_time,
				recovery_bottom,
				recovery_value,
				color=color,
				linestyle=":",
				linewidth=1.15,
				alpha=0.8,
				zorder=1,
			)
	if ylim is not None:
		ax.set_ylim(ylim)
	if xscale == "log" or yscale == "log":
		add_slope_one_guide(ax)
	ax.set_xlabel("Time (s)")
	ax.set_ylabel(r"$\lambda^{\min}$")
	ax.grid(True, which="both", linestyle=DASHED_LINESTYLE, alpha=0.25)
	sns.despine(ax=ax)

	add_split_line_legends(
		fig,
		group_order,
		palette,
		present_groups,
		present_kinetics,
		group_title="Social Interaction",
		kinetics_linestyles=KINETICS_PLOT_LINESTYLES,
	)
	fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
	save_paper_figure(fig, output_path)


def plot_for_setting(
	setting_dir: Path,
	records: list[PaperRecord],
	_args,
	*,
	figures_dir: Path | None = None,
	split_kr2021_regimes: bool = True,
) -> None:
	filtered = [record for record in records if record.panel == MAIN_TEXT_NUM_AGENT]
	if not filtered:
		print(f"[{setting_dir.name}] No N={MAIN_TEXT_NUM_AGENT} data for minimum-eigenvalue plot; skipping.")
		return

	output_dir = (figures_dir or setting_dir / "figures") / "min_eigenvalue_timeseries"
	plot_min_eigenvalue_timeseries_panel(
		filtered,
		MIN_EIGENVALUE_TOP_MODELS,
		output_dir / "identifiable.pdf",
		xscale="log",
		yscale="log",
	)
	tk2020_records = [record for record in filtered if record.group == "TK2020"]
	kr2021_records = [record for record in filtered if record.group == "KR2021"]
	if split_kr2021_regimes:
		kr2021_regimes = split_kr2021_regimes(kr2021_records)
		panels = [
			(
				tk2020_records + kr2021_regimes["fluctuation"],
				MIN_EIGENVALUE_BOTTOM_MODELS,
				"unidentifiable.pdf",
				UNIDENTIFIABLE_YLIM,
			),
			(kr2021_regimes["linear"], ["KR2021"], "linear_KR2021.pdf", None),
		]
	else:
		panels = [
			(tk2020_records, ["TK2020"], "unidentifiable.pdf", UNIDENTIFIABLE_YLIM),
			(kr2021_records, ["KR2021"], "KR2021.pdf", None),
		]
	for panel_records, group_order, filename, ylim in panels:
		if not panel_records:
			print(f"[{setting_dir.name}] No samples for {filename}; skipping.")
			continue
		plot_min_eigenvalue_timeseries_panel(
			panel_records,
			group_order,
			output_dir / filename,
			xscale="linear",
			yscale="linear",
			ylim=ylim,
		)


if __name__ == "__main__":
	run_for_settings(
		plot_for_setting,
		description="Identifiable vs unidentifiable min-eigenvalue time series.",
	)
