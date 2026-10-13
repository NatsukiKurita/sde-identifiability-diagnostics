#!/usr/bin/env python3
"""Fig 3: 2D scatter of estimates on the degenerate parameter pair.

For each unidentifiable model (TK2020, KR2021) the optimum of every N=2 sample is
plotted on the two parameters involved in the degeneracy, together with the
theoretical degeneracy manifold and the ground truth.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

from _common import (
	DASHED_LINESTYLE,
	DEGENERACY_PAIRS,
	KINETICS_LABELS,
	KINETICS_MARKERS,
	MAIN_TEXT_NUM_AGENT,
	PARAMETER_DISPLAY_LABELS,
	PaperRecord,
	_load_truth_params,
	model_palette,
	nested_param_value,
	ordered_kinetics_from_set,
	read_json,
	run_for_settings,
	save_paper_figure,
)

FIGSIZE = (5.6, 5.0)
AXIS_MARGIN_FRACTION = 0.08
MARKER_SIZE = 28.0


def _optimum_pair(record: PaperRecord, x_key: str, y_key: str) -> tuple[float, float] | None:
	summary = read_json(record.result_dir / "summary.json")
	best_params = summary.get("best_params", {})
	x_value = nested_param_value(best_params, x_key)
	y_value = nested_param_value(best_params, y_key)
	if x_value is None or y_value is None:
		return None
	return x_value, y_value


def _manifold_curve(relation: str, constant: float, x_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	if relation == "difference":
		return x_values, x_values - constant
	if relation == "product":
		x_values = x_values[x_values > 0.0]
		return x_values, constant / x_values
	raise ValueError(f"Unknown degeneracy relation: {relation}")


def plot_degeneracy_scatter(records: list[PaperRecord], model: str, output_path: Path) -> None:
	x_key, y_key, relation = DEGENERACY_PAIRS[model]
	cell = sorted(
		[
			record
			for record in records
			if record.panel == MAIN_TEXT_NUM_AGENT and record.group == model and record.result_dir is not None
		],
		key=lambda record: record.base.sample_name,
	)
	if not cell:
		print(f"[{output_path.parent.parent.parent.name}] No N={MAIN_TEXT_NUM_AGENT} data for {model}; skipping.")
		return

	truth_params = _load_truth_params(cell[0].base.setting, cell[0].base.model_id)
	x_true = nested_param_value(truth_params or {}, x_key)
	y_true = nested_param_value(truth_params or {}, y_key)
	if x_true is None or y_true is None:
		print(f"No ground truth for {model} ({x_key}, {y_key}); skipping.")
		return
	constant = x_true - y_true if relation == "difference" else x_true * y_true

	color = model_palette([model])[model]
	points: dict[str, list[tuple[float, float]]] = {}
	for record in cell:
		pair = _optimum_pair(record, x_key, y_key)
		if pair is not None:
			points.setdefault(record.kinetics, []).append(pair)
	if not points:
		print(f"No optimum estimates for {model}; skipping.")
		return
	point_arrays = {kinetics: np.asarray(values, dtype=float) for kinetics, values in points.items()}
	kinetics_order = ordered_kinetics_from_set(set(point_arrays))

	fig, ax = plt.subplots(figsize=FIGSIZE)
	fig.subplots_adjust(left=0.17, right=0.96, bottom=0.14, top=0.80)

	# KR2021 estimates spread over orders of magnitude along the manifold; log axes need positive values.
	all_points = np.vstack([*point_arrays.values(), [[x_true, y_true]]])
	use_log = model != "TK2020" and bool(np.all(all_points > 0.0))
	if use_log:
		ax.set_xscale("log")
		ax.set_yscale("log")
		log_lower = np.log10(all_points.min(axis=0))
		log_upper = np.log10(all_points.max(axis=0))
		log_span = np.maximum(log_upper - log_lower, 0.5)
		lower = 10.0 ** (log_lower - AXIS_MARGIN_FRACTION * log_span)
		upper = 10.0 ** (log_upper + AXIS_MARGIN_FRACTION * log_span)
		grid_x = np.geomspace(lower[0], upper[0], 1024)
	else:
		lower = all_points.min(axis=0)
		upper = all_points.max(axis=0)
		span = np.where(upper > lower, upper - lower, np.maximum(np.abs(upper), 1.0))
		lower = lower - AXIS_MARGIN_FRACTION * span
		upper = upper + AXIS_MARGIN_FRACTION * span
		grid_x = np.linspace(lower[0], upper[0], 1024)
	ax.set_xlim(lower[0], upper[0])
	ax.set_ylim(lower[1], upper[1])

	manifold_x, manifold_y = _manifold_curve(relation, constant, grid_x)
	if use_log:
		positive = manifold_y > 0.0
		manifold_x, manifold_y = manifold_x[positive], manifold_y[positive]
	ax.plot(manifold_x, manifold_y, color="0.45", linestyle="--", linewidth=1.2, zorder=1, scalex=False, scaley=False)

	for kinetics in kinetics_order:
		values = point_arrays[kinetics]
		marker = KINETICS_MARKERS.get(kinetics, "D")
		scatter_kwargs = {"marker": marker, "s": MARKER_SIZE, "linewidths": 1.1, "alpha": 0.75, "zorder": 3}
		if marker == "o":
			scatter_kwargs.update(facecolors="none", edgecolors=color)
		else:
			scatter_kwargs.update(color=color)
		ax.scatter(values[:, 0], values[:, 1], **scatter_kwargs)

	ax.scatter([x_true], [y_true], marker="*", s=180, color="black", zorder=6)

	ax.set_xlabel(PARAMETER_DISPLAY_LABELS.get(x_key, x_key))
	ax.set_ylabel(PARAMETER_DISPLAY_LABELS.get(y_key, y_key))
	ax.grid(True, linestyle=DASHED_LINESTYLE, alpha=0.25)

	manifold_label = (
		rf"{PARAMETER_DISPLAY_LABELS.get(x_key, x_key)} $-$ {PARAMETER_DISPLAY_LABELS.get(y_key, y_key)} = const"
		if relation == "difference"
		else rf"{PARAMETER_DISPLAY_LABELS.get(x_key, x_key)} $\times$ {PARAMETER_DISPLAY_LABELS.get(y_key, y_key)} = const"
	)
	legend_handles = [
		Line2D(
			[0],
			[0],
			color=color,
			linestyle="None",
			marker=KINETICS_MARKERS.get(kinetics, "D"),
			markerfacecolor="none",
			markeredgewidth=1.2,
			markersize=7,
			label=KINETICS_LABELS.get(kinetics, kinetics),
		)
		for kinetics in kinetics_order
	]
	legend_handles.extend(
		[
			Line2D([0], [0], color="black", marker="*", linestyle="None", markersize=11, label="Ground truth"),
			Line2D([0], [0], color="0.45", linestyle="--", linewidth=1.2, label=manifold_label),
		]
	)
	ax.legend(handles=legend_handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
	sns.despine(ax=ax)
	save_paper_figure(fig, output_path)
	plt.close(fig)


def plot_for_setting(setting_dir: Path, records: list[PaperRecord], _args) -> None:
	output_dir = setting_dir / "figures" / "degeneracy_scatter"
	for model in DEGENERACY_PAIRS:
		plot_degeneracy_scatter(records, model, output_dir / f"{model}.pdf")


if __name__ == "__main__":
	run_for_settings(
		plot_for_setting,
		description="2D scatter of estimates on the degenerate parameter pair.",
	)
