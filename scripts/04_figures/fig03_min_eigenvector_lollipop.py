#!/usr/bin/env python3
"""Fig 3: lollipop charts of the minimum-eigenvalue eigenvector."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from _common import (
	DASHED_LINESTYLE,
	EIGENVECTOR_MODELS,
	KINETICS_PLOT_LINESTYLES,
	MAIN_TEXT_NUM_AGENT,
	KINETICS_LABELS,
	PARAMETER_DISPLAY_LABELS,
	RESULT_TIME_SERIES_FILENAME,
	PaperRecord,
	read_json,
	run_for_settings,
	save_paper_figure,
)

# Width grows with the number of parameters; height can be tuned here.
FIGSIZE_HEIGHT = 7.0
FIGSIZE_WIDTH_PER_PARAM = 1.1
FIGSIZE_WIDTH_MIN = 8.5


def _eigenvector_rows(records: list[PaperRecord], model: str) -> list[dict[str, float | str]]:
	rows: list[dict[str, float | str]] = []
	representative_records: list[PaperRecord] = []
	for kinetics in ("Overdamped", "Inertial"):
		candidates = sorted(
			[
				record
				for record in records
				if record.panel == MAIN_TEXT_NUM_AGENT
				and record.group == model
				and record.kinetics == kinetics
				and record.result_dir is not None
			],
			key=lambda record: record.base.sample_name,
		)
		if candidates:
			representative_records.append(candidates[0])

	for record in representative_records:
		results_h5_path = record.result_dir / RESULT_TIME_SERIES_FILENAME
		summary_path = record.result_dir / "summary.json"
		if not results_h5_path.is_file() or not summary_path.is_file():
			continue
		summary_payload = read_json(summary_path)
		parameter_mapping = summary_payload.get("parameter_mapping", [])
		with h5py.File(results_h5_path, "r") as f:
			final_hessian = np.asarray(f["results/optimum/hessians"][-1], dtype=float)
		if final_hessian.ndim != 2 or final_hessian.shape[0] != len(parameter_mapping):
			continue
		_, eigenvectors = np.linalg.eigh(final_hessian)
		min_eigenvector = eigenvectors[:, 0]
		for parameter_idx, (parameter, component) in enumerate(zip(parameter_mapping, min_eigenvector)):
			if not np.isfinite(component):
				continue
			rows.append(
				{
					"Parameter": str(parameter),
					"ParameterIndex": int(parameter_idx),
					"Component": float(component),
					"Dynamics": KINETICS_LABELS.get(record.kinetics, record.kinetics),
				}
			)
	return rows


def plot_min_eigenvector_components(records: list[PaperRecord], model: str, output_path: Path) -> None:
	df = pd.DataFrame(_eigenvector_rows(records, model))
	if df.empty:
		print(f"[{output_path.parent.parent.name}] No eigenvector data for {model}; skipping.")
		return

	fig, ax = plt.subplots(figsize=(max(FIGSIZE_WIDTH_MIN, FIGSIZE_WIDTH_PER_PARAM * df["Parameter"].nunique()), FIGSIZE_HEIGHT))
	min_parameter_indices = df.groupby("Parameter")["ParameterIndex"].min().to_dict()
	parameter_order = sorted(
		min_parameter_indices,
		key=lambda parameter: (min_parameter_indices[parameter], parameter),
	)
	x_positions = {parameter: idx for idx, parameter in enumerate(parameter_order)}
	dynamics_order = [
		label
		for label in (KINETICS_LABELS["Overdamped"], KINETICS_LABELS["Inertial"])
		if label in set(df["Dynamics"])
	]
	palette = dict(zip(dynamics_order, sns.color_palette("deep", n_colors=max(1, len(dynamics_order)))))
	markers = {
		KINETICS_LABELS["Overdamped"]: "o",
		KINETICS_LABELS["Inertial"]: "x",
	}
	offsets = np.linspace(-0.12, 0.12, max(1, len(dynamics_order)))
	for dynamics_idx, dynamics in enumerate(dynamics_order):
		subset = df[df["Dynamics"] == dynamics].sort_values("ParameterIndex")
		x_values = np.asarray([x_positions[parameter] + offsets[dynamics_idx] for parameter in subset["Parameter"]])
		y_values = subset["Component"].to_numpy(dtype=float)
		linewidth = 1.5 if dynamics_idx == 0 else 1.15
		markersize = 6.2 if dynamics_idx == 0 else 5.4
		linestyle = KINETICS_PLOT_LINESTYLES.get(
			"Overdamped" if dynamics == KINETICS_LABELS["Overdamped"] else "Inertial",
			"-",
		)
		ax.vlines(
			x_values,
			0.0,
			y_values,
			color=palette.get(dynamics, "#777777"),
			linestyles=linestyle,
			linewidth=linewidth,
			zorder=2 + dynamics_idx,
		)
		color = palette.get(dynamics, "#777777")
		marker = markers.get(dynamics, "D")
		scatter_kwargs = {
			"marker": marker,
			"s": markersize**2,
			"label": dynamics,
			"zorder": 3 + dynamics_idx,
			"linewidths": 1.2,
		}
		if marker == "o":
			scatter_kwargs.update(facecolors="none", edgecolors=color)
		else:
			scatter_kwargs.update(color=color)
		ax.scatter(x_values, y_values, **scatter_kwargs)
	ax.axhline(0.0, color="0.35", linestyle=DASHED_LINESTYLE, linewidth=0.9, zorder=1)
	inv_sqrt_two = 1.0 / np.sqrt(2.0)
	for value, label in (
		(inv_sqrt_two, r"$1/\sqrt{2}$"),
		(-inv_sqrt_two, r"$-1/\sqrt{2}$"),
	):
		ax.axhline(value, color="0.68", linestyle="-", linewidth=0.9, zorder=1)
		ax.text(
			1.01,
			value,
			label,
			color="0.50",
			va="center",
			ha="left",
			fontsize=9,
			transform=ax.get_yaxis_transform(),
		)
	ax.set_xticks(list(x_positions.values()))
	ax.set_xticklabels([PARAMETER_DISPLAY_LABELS.get(parameter, parameter) for parameter in parameter_order])
	ax.set_xlabel("Parameter")
	ax.set_ylabel("Eigenvector component", labelpad=12)
	ax.tick_params(axis="x", rotation=35)
	for label in ax.get_xticklabels():
		label.set_ha("right")
	ax.margins(x=0.08)
	ax.grid(True, axis="y", linestyle=DASHED_LINESTYLE, alpha=0.25)
	legend_handles = [
		Line2D(
			[0],
			[0],
			color=palette.get(dynamics, "#777777"),
			linestyle=KINETICS_PLOT_LINESTYLES.get(
				"Overdamped" if dynamics == KINETICS_LABELS["Overdamped"] else "Inertial",
				"-",
			),
			marker=markers.get(dynamics, "D"),
			markerfacecolor="none",
			markeredgewidth=1.2,
			linewidth=1.5,
			markersize=6,
			label=dynamics,
		)
		for dynamics in dynamics_order
	]
	ax.legend(handles=legend_handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2)
	sns.despine(ax=ax)
	fig.subplots_adjust(left=0.22, right=0.96, bottom=0.28, top=0.84)
	save_paper_figure(fig, output_path)


def plot_for_setting(setting_dir: Path, records: list[PaperRecord], _args) -> None:
	output_dir = setting_dir / "figures" / "min_eigenvector_components"
	for model in EIGENVECTOR_MODELS:
		plot_min_eigenvector_components(
			records,
			model,
			output_dir / f"{model}.pdf",
		)


if __name__ == "__main__":
	run_for_settings(
		plot_for_setting,
		description="Lollipop charts of minimum-eigenvalue eigenvectors.",
	)
