#!/usr/bin/env python3
"""Fig 3: per-parameter contributions (squared components) to the minimum-eigenvalue eigenvector."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _common import (
	DASHED_LINESTYLE,
	EIGENVECTOR_MODELS,
	MAIN_TEXT_NUM_AGENT,
	KINETICS_LABELS,
	PARAMETER_DISPLAY_LABELS,
	RESULT_TIME_SERIES_FILENAME,
	PaperRecord,
	read_json,
	run_for_settings,
	save_paper_figure,
)

# Height grows with the number of parameters; width can be tuned here.
FIGSIZE_WIDTH = 6.4
FIGSIZE_HEIGHT_PER_PARAM = 0.7
FIGSIZE_HEIGHT_MIN = 4.5


def _eigenvector_rows(records: list[PaperRecord], model: str) -> list[dict[str, float | str]]:
	rows: list[dict[str, float | str]] = []
	selected_records = [
		record
		for record in records
		if record.panel == MAIN_TEXT_NUM_AGENT
		and record.group == model
		and record.kinetics in ("Overdamped", "Inertial")
		and record.result_dir is not None
	]

	for record in selected_records:
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
					"Sample": record.base.sample_name,
					"Parameter": str(parameter),
					"ParameterIndex": int(parameter_idx),
					"Contribution": float(component) ** 2,
					"Dynamics": KINETICS_LABELS.get(record.kinetics, record.kinetics),
				}
			)
	return rows


def plot_min_eigenvector_components(records: list[PaperRecord], model: str, output_path: Path) -> None:
	df = pd.DataFrame(_eigenvector_rows(records, model))
	if df.empty:
		print(f"[{output_path.parent.parent.name}] No eigenvector data for {model}; skipping.")
		return

	min_parameter_indices = df.groupby("Parameter")["ParameterIndex"].min().to_dict()
	parameter_order = sorted(
		min_parameter_indices,
		key=lambda parameter: (min_parameter_indices[parameter], parameter),
	)
	figsize = (FIGSIZE_WIDTH, max(FIGSIZE_HEIGHT_MIN, FIGSIZE_HEIGHT_PER_PARAM * len(parameter_order)))
	fig, ax = plt.subplots(figsize=figsize)
	y_positions = {parameter: idx for idx, parameter in enumerate(parameter_order)}
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
	offsets = np.linspace(-0.15, 0.15, max(1, len(dynamics_order)))
	for dynamics_idx, dynamics in enumerate(dynamics_order):
		stats = (
			df[df["Dynamics"] == dynamics]
			.groupby("Parameter")["Contribution"]
			.agg(["mean", "std"])
			.reindex(parameter_order)
			.dropna(subset=["mean"])
		)
		y_values = np.asarray([y_positions[parameter] + offsets[dynamics_idx] for parameter in stats.index])
		color = palette.get(dynamics, "#777777")
		marker = markers.get(dynamics, "D")
		ax.errorbar(
			stats["mean"].to_numpy(dtype=float),
			y_values,
			xerr=stats["std"].fillna(0.0).to_numpy(dtype=float),
			fmt=marker,
			color=color,
			markerfacecolor="none" if marker == "o" else color,
			markeredgewidth=1.2,
			markersize=6,
			elinewidth=1.2,
			capsize=3,
			linestyle="none",
			label=dynamics,
			zorder=3 + dynamics_idx,
		)
	ax.axvline(0.5, color="0.68", linestyle="-", linewidth=0.9, zorder=1)
	ax.text(
		0.5,
		1.01,
		"1/2",
		color="0.50",
		va="bottom",
		ha="center",
		fontsize=9,
		transform=ax.get_xaxis_transform(),
	)
	ax.set_yticks(list(y_positions.values()))
	ax.set_yticklabels([PARAMETER_DISPLAY_LABELS.get(parameter, parameter) for parameter in parameter_order])
	ax.invert_yaxis()
	ax.set_xlim(-0.03, 1.0)
	ax.set_xlabel("Contribution to minimum-eigenvalue eigenvector")
	ax.set_ylabel("Parameter")
	ax.margins(y=0.08)
	ax.grid(True, axis="x", linestyle=DASHED_LINESTYLE, alpha=0.25)
	ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2)
	sns.despine(ax=ax)
	fig.tight_layout()
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
		description="Contributions to minimum-eigenvalue eigenvectors.",
	)
