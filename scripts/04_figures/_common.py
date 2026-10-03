#!/usr/bin/env python3
"""Shared helpers for social-interaction paper figures."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns
import toml
from src.constants.simulation_01 import SocialSimConfig
from src.figure_config import set_paper_style
from src.parameter_estimation.builder import extract_params_from_toml
from src.paths import DATA_ROOT, RESULTS_ROOT

PARAMETER_EPS = 1e-12
KINETICS_MARKERS = {
	"Overdamped": "o",
	"Inertial": "x",
}
KINETICS_LINESTYLES = {
	"Overdamped": "-",
	"Inertial": "--",
}
KINETICS_LABELS = {
	"Overdamped": "Overdamped",
	"Inertial": "Inertial",
}
_COLORBLIND = sns.color_palette("colorblind", n_colors=10)
SOCIAL_MODEL_COLORS = {
	"G2012": _COLORBLIND[0],
	"KR2021": _COLORBLIND[1],
	"KV": _COLORBLIND[2],
	"Null": _COLORBLIND[7],
	"TK2020": _COLORBLIND[4],
}
MARKER_GROUP_LEGEND_MAX_NCOLS = 2
MARKER_GROUP_LEGEND_COLUMNSPACING = 1.6
SPLIT_MARKER_GROUP_BBOX = (0.01, 1.0)
SPLIT_MARKER_DYNAMICS_BBOX = (0.99, 1.0)


@dataclass(frozen=True)
class PaperRecord:
	base: Any
	group: str
	kinetics: str
	panel: int | str
	max_abs_log_parameter_error: float
	burn_in_start_index: int
	result_dir: Path | None = None


def read_json(path: Path) -> dict:
	with open(path, "r") as f:
		return json.load(f)


def nested_param_value(param_tree: dict, path: str) -> float | None:
	module, key = path.split(".", 1)
	if module not in param_tree or key not in param_tree[module]:
		return None
	value = param_tree[module][key]
	if isinstance(value, bool) or not isinstance(value, (int, float)):
		return None
	value_float = float(value)
	return value_float if math.isfinite(value_float) else None


def compute_max_abs_log_parameter_error(
	parameter_mapping: list[str],
	best_params: dict,
	truth_params: dict,
) -> float | None:
	errors: list[float] = []
	for path in parameter_mapping:
		opt_value = nested_param_value(best_params, path)
		true_value = nested_param_value(truth_params, path)
		if opt_value is None or true_value is None:
			continue
		if abs(true_value) < PARAMETER_EPS:
			continue
		ratio = opt_value / true_value
		if ratio > 0.0 and math.isfinite(ratio):
			errors.append(abs(math.log(ratio)))

	if not errors:
		return None
	return float(max(errors))


def model_palette(group_order: list[str]) -> dict[str, tuple[float, float, float]]:
	fallback_colors = iter(sns.color_palette("colorblind", n_colors=max(1, len(group_order))))
	return {
		group: SOCIAL_MODEL_COLORS.get(group) or next(fallback_colors)
		for group in group_order
	}


def marker_group_legend_ncols(n_groups: int) -> int:
	return min(MARKER_GROUP_LEGEND_MAX_NCOLS, max(1, n_groups))


def ordered_kinetics_from_set(kinetics_values: set[str]) -> list[str]:
	order = [kinetics for kinetics in ("Overdamped", "Inertial") if kinetics in kinetics_values]
	order.extend(sorted(kinetics_values - set(order)))
	return order


def _draw_axis_guide(
	ax: plt.Axes,
	x_values: np.ndarray,
	y_values: np.ndarray,
	label: str,
	label_x: float,
	label_y: float,
) -> None:
	ax.plot(
		x_values,
		y_values,
		color="0.55",
		linestyle=":",
		linewidth=1.0,
		zorder=0,
		label="_nolegend_",
		scalex=False,
		scaley=False,
	)
	ax.text(
		label_x,
		label_y,
		label,
		color="0.45",
		fontsize=9,
		ha="center",
		va="bottom",
	)


def add_slope_one_guide(
	ax: plt.Axes,
	*,
	x_start: float = 2.0,
	x_end: float = 10.0,
	y_top_divisor: float = 1.5,
) -> None:
	y_bottom, y_top = ax.get_ylim()
	if ax.get_yscale() == "log" or ax.get_xscale() == "log":
		guide_end = y_top / y_top_divisor
		guide_scale = guide_end / x_end
		x_values = np.asarray([x_start, x_end])
		y_values = guide_scale * x_values
		label_x = float(np.sqrt(x_start * x_end))
		label_y = guide_scale * label_x * 1.35
	else:
		y_range = y_top - y_bottom
		guide_end = y_bottom + 0.82 * y_range
		intercept = guide_end - x_end
		x_values = np.asarray([x_start, x_end])
		y_values = x_values + intercept
		label_x = 0.5 * (x_start + x_end)
		label_y = label_x + intercept + 0.05 * y_range

	_draw_axis_guide(ax, x_values, y_values, "slope=1", label_x, label_y)


def add_sqrt_x_log_x_guide(
	ax: plt.Axes,
	*,
	x_start: float = 2.0,
	x_end: float = 10.0,
	y_top_divisor: float = 1.5,
) -> None:
	_, y_top = ax.get_ylim()
	x_min = max(float(x_start), np.nextafter(1.0, 2.0))
	x_values = np.unique(
		np.concatenate(
			[
				np.geomspace(x_min, x_end, 512),
				np.linspace(x_min, min(x_end, max(x_min * 3.0, x_min + 2.0)), 256),
			]
		)
	)
	x_values = x_values[(x_values >= x_min) & (x_values <= x_end)]
	shape = np.sqrt(x_values) * np.log(x_values)
	positive = shape > 0.0
	x_values = x_values[positive]
	shape = shape[positive]
	if x_values.size < 2:
		return
	guide_end = y_top / y_top_divisor
	guide_scale = guide_end / float(shape[-1])
	y_values = guide_scale * shape
	label_x = float(np.sqrt(x_min * x_end))
	label_y = guide_scale * np.sqrt(label_x) * np.log(label_x) * 1.35
	_draw_axis_guide(ax, x_values, y_values, r"$\sqrt{T}\,\log T$", label_x, label_y)


def add_split_line_legends(
	fig: plt.Figure,
	group_order: list[str],
	palette: dict[str, tuple[float, float, float]],
	present_groups: set[str],
	present_kinetics: set[str],
	*,
	group_title: str,
	kinetics_linestyles: dict[str, Any],
	kinetics_style: str = "line",
	group_ncols: int | None = None,
) -> None:
	group_handles = [
		Line2D(
			[0],
			[0],
			color=palette.get(group, "#777777"),
			linewidth=2.0,
			label=group,
		)
		for group in group_order
		if group in present_groups
	]
	kinetics_handles = [
		Line2D(
			[0],
			[0],
			color="0.25",
			linestyle="None" if kinetics_style == "marker" else kinetics_linestyles.get(kinetics, ":"),
			marker=None if kinetics_style == "line" else KINETICS_MARKERS.get(kinetics, "D"),
			linewidth=2.0,
			markersize=6,
			markerfacecolor="none",
			markeredgewidth=1.2,
			label=KINETICS_LABELS.get(kinetics, kinetics),
		)
		for kinetics in ordered_kinetics_from_set(present_kinetics)
	]
	if group_handles:
		resolved_group_ncols = group_ncols
		if resolved_group_ncols is None:
			if kinetics_style == "marker":
				resolved_group_ncols = marker_group_legend_ncols(len(group_handles))
			else:
				resolved_group_ncols = 2 if len(group_handles) > 4 else min(3, len(group_handles))
		fig.legend(
			handles=group_handles,
			title=group_title,
			loc="upper left",
			ncol=resolved_group_ncols,
			bbox_to_anchor=SPLIT_MARKER_GROUP_BBOX,
			frameon=False,
			columnspacing=MARKER_GROUP_LEGEND_COLUMNSPACING,
		)
	if kinetics_handles:
		fig.legend(
			handles=kinetics_handles,
			title="Dynamics",
			loc="upper right",
			bbox_to_anchor=SPLIT_MARKER_DYNAMICS_BBOX,
			frameon=False,
			borderaxespad=0.0,
		)


DEFAULT_NUM_AGENTS = list(SocialSimConfig.NUM_AGENTS)
MAIN_TEXT_NUM_AGENTS = [2]
MAIN_TEXT_NUM_AGENT = 2
EIGENVECTOR_MODELS = ["KR2021", "TK2020"]
MIN_EIGENVALUE_TOP_MODELS = ["Null", "G2012", "KV"]
MIN_EIGENVALUE_BOTTOM_MODELS = ["KR2021", "TK2020"]
PARAMETER_ACCURACY_MODELS = ["Null", "G2012", "KV"]
PARAMETER_ACCURACY_KINETICS = "Inertial"
ERROR_PLOT_KINETICS = "Overdamped"
SPECIAL_MIN_EIGENVALUE_SERIES = {
	("KR2021", "Inertial"): "results_0006_N2_0020",
}
MIN_EIGENVALUE_PLOT_SECONDS = 100.0
MIN_EIGENVALUE_PLOT_INTERVAL_SECONDS = 1/24
MIN_EIGENVALUE_LOG_FLOOR = 1e-12
SPAGHETTI_XSCALE = "log"
SPAGHETTI_YSCALE = "symlog"
# |value| < linthresh is drawn linearly on spaghetti y axes.
SPAGHETTI_SYMLOG_LINTHRESH_Y = 1
SPAGHETTI_SYMLOG_LINSCALE_Y = 0.7
SPAGHETTI_GUIDE_X_START = 5.0
SPAGHETTI_GUIDE_X_END = 100.0
SPAGHETTI_GUIDE_Y_TOP_DIVISOR = 1.5
DASHED_LINESTYLE = (0, (2.0, 0.8))
KINETICS_PLOT_LINESTYLES = {
	**KINETICS_LINESTYLES,
	"Inertial": DASHED_LINESTYLE,
}
PARAMETER_DISPLAY_LABELS = {
	"Kinetics.sigma_omega": r"$\sigma_{\omega}$",
	"Kinetics.sigma_theta": r"$\sigma_{\theta}$",
	"Kinetics.tau": r"$\tau$",
	"Social.k_alg": r"$k^{\mathrm{alg}}$",
	"Social.k_pos": r"$k^{\mathrm{pos}}$",
	"Social.m": r"$m$",
	"Social.d_eq": r"$d^{\mathrm{eq}}$",
	"Social.k_coh": r"$k^{\mathrm{coh}}$",
	"Social.k_sep": r"$k^{\mathrm{sep}}$",
}
ALL_SOCIAL_MODELS = ["Null", "G2012", "KV", "KR2021", "TK2020"]
RELATIVE_ERROR_FLOOR = 1e-12


@dataclass(frozen=True)
class SampleRecord:
	setting: str
	model_id: str
	model_label: str
	num_agents: int
	sample_name: str
	time_seconds: list[float]
	min_eigenvalues: list[float]
	min_relative_error: float
	max_relative_error: float


def _configure_plot_style() -> None:
	sns.set_theme(style="ticks", context="paper", font_scale=1.3)
	plt.rcParams["font.family"] = "sans-serif"
	plt.rcParams["axes.linewidth"] = 1.2


def _short_model_name(combo_key: str) -> str:
	parts = combo_key.split("-")
	if len(parts) >= 2:
		kinetics = parts[0]
		social = parts[1].replace("Soc_", "")
		return f"{kinetics}/{social}"
	return combo_key


def _split_model_label(model_label: str) -> tuple[str, str]:
	if "/" not in model_label:
		return "Other", model_label
	kinetics, model = model_label.split("/", 1)
	return kinetics, model


def _load_ledger_inverse(setting_name: str) -> dict[int, str]:
	ledger_path = DATA_ROOT / "simulation" / setting_name / "ledger.json"
	if not ledger_path.is_file():
		return {}
	with open(ledger_path, "r") as f:
		ledger = json.load(f)
	return {int(model_id): combo_key for combo_key, model_id in ledger.items()}


def _model_label(model_id: str, ledger_inverse: dict[int, str]) -> str:
	model_num = int(model_id)
	if model_num in ledger_inverse:
		return _short_model_name(ledger_inverse[model_num])
	return f"model_{model_id}"


def _parse_result_paths(summary_path: Path) -> tuple[str, str, int, str] | None:
	sample_dir = summary_path.parent
	agent_dir = sample_dir.parent.name
	model_dir = sample_dir.parent.parent.name
	setting_name = sample_dir.parent.parent.parent.name
	sample_name = sample_dir.name

	if not model_dir.startswith("model_") or not agent_dir.startswith("N"):
		return None

	model_id = model_dir.split("_", 1)[1]
	num_agents = int(agent_dir[1:])
	return setting_name, model_id, num_agents, sample_name


def _compute_parameter_ratio_bounds(
	parameter_mapping: list[str],
	best_params: dict,
	truth_params: dict,
) -> tuple[float, float] | None:
	ratios: list[float] = []
	for path in parameter_mapping:
		module, key = path.split(".", 1)
		if module not in best_params or key not in best_params[module]:
			continue
		if module not in truth_params or key not in truth_params[module]:
			continue

		theta_opt = float(best_params[module][key])
		theta_true = float(truth_params[module][key])
		if abs(theta_true) < RELATIVE_ERROR_FLOOR:
			continue

		ratio = abs(theta_opt / theta_true)
		if math.isfinite(ratio) and ratio > 0.0:
			ratios.append(ratio)

	if not ratios:
		return None
	return float(min(ratios)), float(max(ratios))


def _load_truth_params(setting_name: str, model_id: str) -> dict | None:
	config_path = (
		DATA_ROOT
		/ "simulation"
		/ setting_name
		/ f"model_{model_id}"
		/ f"config_{model_id}.toml"
	)
	if not config_path.is_file():
		return None
	with open(config_path, "r") as f:
		config_toml = toml.load(f)
	return extract_params_from_toml(config_toml)


def _load_sample_record(summary_path: Path, ledger_inverse: dict[int, str]) -> SampleRecord | None:
	if (summary_path.parent / "skip_message.json").is_file():
		return None

	parsed = _parse_result_paths(summary_path)
	if parsed is None:
		return None
	setting_name, model_id, num_agents, sample_name = parsed

	eval_path = summary_path.parent / "eval_trend_at_opt.json"
	if not eval_path.is_file():
		return None

	with open(summary_path, "r") as f:
		summary_payload = json.load(f)
	with open(eval_path, "r") as f:
		eval_payload = json.load(f)

	time_seconds = [float(x) for x in eval_payload.get("time_seconds", [])]
	min_eigenvalues = [float(x) for x in eval_payload.get("min_eigenvalues", [])]
	if len(time_seconds) != len(min_eigenvalues) or not time_seconds:
		return None

	truth_params = _load_truth_params(setting_name, model_id)
	if truth_params is None:
		return None

	error_bounds = _compute_parameter_ratio_bounds(
		summary_payload.get("parameter_mapping", []),
		summary_payload.get("best_params", {}),
		truth_params,
	)
	if error_bounds is None:
		return None
	min_relative_error, max_relative_error = error_bounds

	return SampleRecord(
		setting=setting_name,
		model_id=model_id,
		model_label=_model_label(model_id, ledger_inverse),
		num_agents=num_agents,
		sample_name=sample_name,
		time_seconds=time_seconds,
		min_eigenvalues=min_eigenvalues,
		min_relative_error=min_relative_error,
		max_relative_error=max_relative_error,
	)


def save_paper_figure(fig: plt.Figure, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, format="pdf")
	fig.savefig(output_path.with_suffix(".png"), format="png")
	print(f"Saved {output_path}")


def collect_paper_records(setting_dir: Path) -> list[PaperRecord]:
	setting_name = setting_dir.name
	ledger_inverse = _load_ledger_inverse(setting_name)
	records: list[PaperRecord] = []

	for summary_path in sorted(setting_dir.glob("model_*/N*/results_*/summary.json")):
		base_record = _load_sample_record(summary_path, ledger_inverse)
		if base_record is None:
			continue

		summary_payload = read_json(summary_path)
		truth_params = _load_truth_params(setting_name, base_record.model_id)
		if truth_params is None:
			continue

		max_abs_log_error = compute_max_abs_log_parameter_error(
			summary_payload.get("parameter_mapping", []),
			summary_payload.get("best_params", {}),
			truth_params,
		)
		if max_abs_log_error is None:
			continue

		eval_payload = read_json(summary_path.parent / "eval_trend_at_opt.json")
		kinetics, social_module = _split_model_label(base_record.model_label)
		records.append(
			PaperRecord(
				base=base_record,
				group=social_module,
				kinetics=kinetics,
				panel=base_record.num_agents,
				max_abs_log_parameter_error=max_abs_log_error,
				burn_in_start_index=max(0, int(eval_payload.get("evaluation", {}).get("burn_in_start_index", 0))),
				result_dir=summary_path.parent,
			)
		)

	return records


def record_min_eigenvalue_series(record: PaperRecord) -> tuple[np.ndarray, np.ndarray]:
	times = np.asarray(record.base.time_seconds, dtype=float)
	values = np.asarray(record.base.min_eigenvalues, dtype=float)
	finite_mask = np.isfinite(times) & np.isfinite(values)
	return times[finite_mask], values[finite_mask]


def downsample_min_eigenvalue_series(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	if times.size < 2:
		return times, values
	start = np.ceil(times[0] / MIN_EIGENVALUE_PLOT_INTERVAL_SECONDS) * MIN_EIGENVALUE_PLOT_INTERVAL_SECONDS
	end = np.floor(times[-1] / MIN_EIGENVALUE_PLOT_INTERVAL_SECONDS) * MIN_EIGENVALUE_PLOT_INTERVAL_SECONDS
	sampled_times = np.arange(
		start,
		end + MIN_EIGENVALUE_PLOT_INTERVAL_SECONDS * 0.5,
		MIN_EIGENVALUE_PLOT_INTERVAL_SECONDS,
	)
	if sampled_times.size < 2:
		return times, values
	return sampled_times, np.interp(sampled_times, times, values)


def mean_series(series: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
	series = [(times, values) for times, values in series if times.size >= 2 and values.size >= 2]
	if not series:
		return None

	start = max(float(times[0]) for times, _ in series)
	end = min(float(times[-1]) for times, _ in series)
	if end <= start:
		return None

	reference_times = max((times for times, _ in series), key=len)
	grid = reference_times[(reference_times >= start) & (reference_times <= end)]
	if grid.size < 2:
		grid = np.linspace(start, end, 100)

	interpolated = [np.interp(grid, times, values) for times, values in series]
	return grid, np.mean(np.vstack(interpolated), axis=0)


def mean_std_series(
	series: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
	series = [(times, values) for times, values in series if times.size >= 2 and values.size >= 2]
	if not series:
		return None

	start = max(float(times[0]) for times, _ in series)
	end = min(float(times[-1]) for times, _ in series)
	if end <= start:
		return None

	reference_times = max((times for times, _ in series), key=len)
	grid = reference_times[(reference_times >= start) & (reference_times <= end)]
	if grid.size < 2:
		grid = np.linspace(start, end, 100)

	stacked = np.vstack([np.interp(grid, times, values) for times, values in series])
	mean_values = np.mean(stacked, axis=0)
	if stacked.shape[0] > 1:
		std_values = np.std(stacked, axis=0, ddof=1)
	else:
		std_values = np.zeros_like(mean_values)
	return grid, mean_values, std_values


def plot_mean_std_band(
	ax: plt.Axes,
	times: np.ndarray,
	mean_values: np.ndarray,
	std_values: np.ndarray,
	color,
	*,
	linestyle: str | tuple = "-",
	marker: str | None = "o",
	band_alpha: float = 0.22,
	clip_lower: float | None = None,
) -> None:
	lower = mean_values - std_values
	upper = mean_values + std_values
	if clip_lower is not None:
		lower = np.maximum(clip_lower, lower)
	ax.fill_between(times, lower, upper, color=color, alpha=band_alpha, linewidth=0, zorder=1)
	ax.plot(
		times,
		mean_values,
		color=color,
		linestyle=linestyle,
		linewidth=1.9,
		marker=marker,
		markersize=4.5,
		zorder=2,
	)


def prepare_min_eigenvalue_line(
	times: np.ndarray,
	values: np.ndarray,
	*,
	xscale: str,
	yscale: str,
) -> tuple[np.ndarray, np.ndarray]:
	finite_mask = np.isfinite(times) & np.isfinite(values) & (times <= MIN_EIGENVALUE_PLOT_SECONDS)
	if xscale == "log":
		finite_mask &= times > 0.0
	times = times[finite_mask]
	values = values[finite_mask]
	if yscale == "log":
		values = np.where(values > MIN_EIGENVALUE_LOG_FLOOR, values, np.nan)
	return times, values


def _apply_axis_scale(ax: plt.Axes, axis: str, scale: str) -> None:
	kwargs: dict[str, float] = {}
	if scale == "symlog":
		kwargs = {
			"linthresh": SPAGHETTI_SYMLOG_LINTHRESH_Y,
			"linscale": SPAGHETTI_SYMLOG_LINSCALE_Y,
		}
	if axis == "x":
		ax.set_xscale(scale, **kwargs)
	else:
		ax.set_yscale(scale, **kwargs)


def style_min_eigenvalue_axis(
	ax: plt.Axes,
	*,
	xscale: str,
	yscale: str,
	x_values: list[np.ndarray],
	y_values: list[np.ndarray],
	identifiable: bool,
) -> None:
	_apply_axis_scale(ax, "x", xscale)
	_apply_axis_scale(ax, "y", yscale)
	if x_values:
		left = max(min(float(np.nanmin(times)) for times in x_values), np.finfo(float).tiny) if xscale == "log" else 0.0
		ax.set_xlim(left=left, right=MIN_EIGENVALUE_PLOT_SECONDS)
	if yscale == "log" and y_values:
		finite_mins = [float(np.nanmin(values)) for values in y_values if np.any(np.isfinite(values))]
		if finite_mins:
			ax.set_ylim(bottom=max(MIN_EIGENVALUE_LOG_FLOOR, min(finite_mins)))
	guide_kwargs = {
		"x_start": SPAGHETTI_GUIDE_X_START,
		"x_end": SPAGHETTI_GUIDE_X_END,
		"y_top_divisor": SPAGHETTI_GUIDE_Y_TOP_DIVISOR,
	}
	if identifiable:
		add_slope_one_guide(ax, **guide_kwargs)
	else:
		add_sqrt_x_log_x_guide(ax, **guide_kwargs)
	ax.set_xlabel("Time (s)")
	ax.set_ylabel(r"$\lambda^{\mathrm{min}}_t$ (Symlog-scaled)")
	ax.grid(True, which="both", linestyle=DASHED_LINESTYLE, alpha=0.25)
	sns.despine(ax=ax)


def plot_spaghetti_min_eigenvalue(
	records: list[PaperRecord],
	output_path: Path,
	*,
	figsize: tuple[float, float],
	color,
	title: str | None = None,
	line_alpha: float = 0.2,
	linewidth: float = 0.3,
) -> None:
	fig, ax = plt.subplots(figsize=figsize)
	x_values: list[np.ndarray] = []
	y_values: list[np.ndarray] = []
	n_lines = 0
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
		ax.plot(times, values, color=color, linestyle="-", linewidth=linewidth, alpha=line_alpha)
		n_lines += 1

	if n_lines == 0:
		print(f"[{output_path.parent.parent.name}] No spaghetti lines for {output_path.name}; skipping.")
		plt.close(fig)
		return

	style_min_eigenvalue_axis(
		ax,
		xscale=SPAGHETTI_XSCALE,
		yscale=SPAGHETTI_YSCALE,
		x_values=x_values,
		y_values=y_values,
		identifiable=records[0].group not in MIN_EIGENVALUE_BOTTOM_MODELS,
	)
	if title:
		ax.set_title(title)
	fig.tight_layout()
	save_paper_figure(fig, output_path)


def configure_plot_style() -> None:
	_configure_plot_style()
	set_paper_style()


def discover_setting_dirs(results_root: Path, setting: str | None) -> list[Path]:
	if not results_root.is_dir():
		return []
	if setting is not None:
		setting_dir = results_root / setting
		return [setting_dir] if setting_dir.is_dir() else []
	return sorted(path for path in results_root.iterdir() if path.is_dir() and path.name.startswith("setting_"))


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
	parser.add_argument(
		"--results-root",
		type=Path,
		default=RESULTS_ROOT,
		help="Root directory containing setting_* result folders.",
	)
	parser.add_argument("--setting", type=str, default=None, help="Process only this setting.")
	parser.add_argument(
		"--num-agents",
		type=int,
		nargs="+",
		default=DEFAULT_NUM_AGENTS,
		help="Population sizes to include in supplementary panels.",
	)
	parser.add_argument(
		"--main-num-agents",
		type=int,
		nargs="+",
		default=MAIN_TEXT_NUM_AGENTS,
		help="Population sizes for main-text panels.",
	)
	return parser


def parse_common_args(description: str) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=description)
	return add_common_arguments(parser).parse_args()


def run_for_settings(plot_setting, args: argparse.Namespace | None = None, *, description: str = "") -> None:
	if args is None:
		args = parse_common_args(description)
	configure_plot_style()
	setting_dirs = discover_setting_dirs(args.results_root, args.setting)
	if not setting_dirs:
		raise FileNotFoundError(f"No setting directories found under {args.results_root}")
	for setting_dir in setting_dirs:
		records = collect_paper_records(setting_dir)
		if not records:
			print(f"No completed paper-figure records found under {setting_dir}; skipping.")
			continue
		print(f"Loaded {len(records)} samples from {setting_dir.name}")
		plot_setting(setting_dir, records, args)
