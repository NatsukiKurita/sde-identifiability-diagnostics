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
	MIN_EIGENVALUE_TOP_MODELS,
	SPECIAL_MIN_EIGENVALUE_SERIES,
	downsample_min_eigenvalue_series,
	mean_series,
	PaperRecord,
	add_slope_one_guide,
	add_split_line_legends,
	model_palette,
	ordered_kinetics_from_set,
	record_min_eigenvalue_series,
	run_for_settings,
	save_paper_figure,
)

# Adjust this figure size independently.
FIGSIZE = (6.4, 4.0)
INSET_START_SECONDS = 10.0
# (kinetics, ylim) per inset, drawn left to right.
BOTTOM_INSET_YLIMS = {
	"TK2020": (
		("Inertial", (-6e-8, 6e-8)),
		("Overdamped", (-1.5e-9, 1.5e-9)),
	),
}
INSET_BOUNDS = (
	(0.12, 0.15, 0.38, 0.5),
	(0.6, 0.15, 0.38, 0.5),
)

def _min_eigenvalue_plot_series(
	records: list[PaperRecord],
	group: str,
	kinetics: str,
) -> tuple[np.ndarray, np.ndarray] | None:
	special_sample_name = SPECIAL_MIN_EIGENVALUE_SERIES.get((group, kinetics))
	if special_sample_name is not None:
		for record in records:
			if record.base.sample_name == special_sample_name:
				series = record_min_eigenvalue_series(record)
				break
		else:
			series = mean_series([record_min_eigenvalue_series(record) for record in records])
	else:
		series = mean_series([record_min_eigenvalue_series(record) for record in records])

	if series is None:
		return None
	return downsample_min_eigenvalue_series(*series)


def _add_zoom_insets(
	ax,
	plotted_lines: list[tuple[str, np.ndarray, np.ndarray, dict]],
	inset_specs: tuple[tuple[str, tuple[float, float]], ...],
) -> None:
	right = ax.get_xlim()[1]
	for bounds, (kinetics, ylim) in zip(INSET_BOUNDS, inset_specs):
		inset = ax.inset_axes(bounds)
		for line_kinetics, times, values, line_kwargs in plotted_lines:
			if line_kinetics == kinetics:
				inset.plot(times, values, **line_kwargs)
		inset.set_xlim(INSET_START_SECONDS, right)
		inset.set_ylim(ylim)
		inset.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
		inset.tick_params(labelsize=7)
		inset.yaxis.get_offset_text().set_fontsize(7)
		inset.axhline(0.0, color="#999999", linewidth=0.6, zorder=0)
		inset.grid(True, linestyle=DASHED_LINESTYLE, alpha=0.25)
		inset.set_title(kinetics, fontsize=8)


def plot_min_eigenvalue_timeseries_panel(
	filtered: list[PaperRecord],
	group_order: list[str],
	output_path: Path,
	*,
	xscale: str,
	yscale: str,
	inset_specs: tuple[tuple[str, tuple[float, float]], ...] | None = None,
) -> None:
	kinetics_order = ordered_kinetics_from_set({record.kinetics for record in filtered})
	palette = model_palette(group_order)

	fig, ax = plt.subplots(figsize=FIGSIZE)
	present_groups: set[str] = set()
	present_kinetics: set[str] = set()
	x_values: list[np.ndarray] = []
	y_values: list[np.ndarray] = []
	recoveries: list[tuple[float, float, tuple[float, float, float]]] = []
	plotted_lines: list[tuple[str, np.ndarray, np.ndarray, dict]] = []
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
			finite_mask = np.isfinite(times) & np.isfinite(values) & (times <= MIN_EIGENVALUE_PLOT_SECONDS)
			if xscale == "log":
				finite_mask &= times > 0.0
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
			x_values.append(times)
			y_values.append(values[np.isfinite(values)])
			linewidth = 1.15
			linestyle = KINETICS_PLOT_LINESTYLES.get(kinetics, ":")
			alpha = 0.9
			if group == "TK2020" and kinetics == "Inertial":
				linewidth = 1.15
				alpha = 0.75
			line_kwargs = dict(
				color=palette.get(group, "#777777"),
				linestyle=linestyle,
				linewidth=linewidth,
				alpha=alpha,
			)
			ax.plot(times, values, **line_kwargs)
			plotted_lines.append((kinetics, times, values, line_kwargs))

	if not present_groups:
		print(f"[{output_path.parent.parent.name}] No minimum-eigenvalue lines to plot; skipping.")
		plt.close(fig)
		return

	ax.set_xscale(xscale)
	ax.set_yscale(yscale)
	if x_values:
		left = max(min(float(np.nanmin(times)) for times in x_values), np.finfo(float).tiny) if xscale == "log" else 0.0
		ax.set_xlim(left=left, right=MIN_EIGENVALUE_PLOT_SECONDS)
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
	if xscale == "log" or yscale == "log":
		add_slope_one_guide(ax)
	ax.set_xlabel("Time (s)")
	ax.set_ylabel(r"$\lambda^{\min}$")
	ax.grid(True, which="both", linestyle=DASHED_LINESTYLE, alpha=0.25)
	sns.despine(ax=ax)
	if inset_specs is not None:
		_add_zoom_insets(ax, plotted_lines, inset_specs)

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


def plot_for_setting(setting_dir: Path, records: list[PaperRecord], _args) -> None:
	filtered = [record for record in records if record.panel == MAIN_TEXT_NUM_AGENT]
	if not filtered:
		print(f"[{setting_dir.name}] No N={MAIN_TEXT_NUM_AGENT} data for minimum-eigenvalue plot; skipping.")
		return

	output_dir = setting_dir / "figures" / "min_eigenvalue_timeseries"
	plot_min_eigenvalue_timeseries_panel(
		filtered,
		MIN_EIGENVALUE_TOP_MODELS,
		output_dir / "min_eigenvalue_timeseries_top.pdf",
		xscale="log",
		yscale="log",
	)
	for group in MIN_EIGENVALUE_BOTTOM_MODELS:
		bottom_filtered = [record for record in filtered if record.group == group]
		if not bottom_filtered:
			print(
				f"[{setting_dir.name}] No N={MAIN_TEXT_NUM_AGENT} {group} data for unidentifiable min-eigenvalue plot; skipping."
			)
			continue
		plot_min_eigenvalue_timeseries_panel(
			bottom_filtered,
			[group],
			output_dir / f"min_eigenvalue_timeseries_bottom_{group}.pdf",
			xscale="linear",
			yscale="linear",
			inset_specs=BOTTOM_INSET_YLIMS.get(group),
		)


if __name__ == "__main__":
	run_for_settings(
		plot_for_setting,
		description="Identifiable vs unidentifiable min-eigenvalue time series.",
	)
