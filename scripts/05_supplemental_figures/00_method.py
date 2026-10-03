#!/usr/bin/env python3
"""Plot minimum-eigenvalue trajectories for selected well-behaved and volatile samples.

Creates one figure with both trajectories from ``eval_trend_at_opt.json`` under each
sample result directory. The stored 24 Hz series is sampled once per second for the
plot. Both axes use linear scales. By default the time axis is limited to 60 s; use
``--plot-to-end`` to show the full trajectory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.paths import PROJECT_ROOT

# Edit these sample IDs to choose representative samples without CLI arguments.
DEFAULT_SETTING = "setting_01_1"
DEFAULT_WELL_BEHAVED_SAMPLE_ID = "0007_N2_0003"
DEFAULT_VOLATILE_SAMPLE_ID = "0009_N2_0003"


def _sample_path_from_id(sample_id: str, setting: str = DEFAULT_SETTING) -> Path:
	model_id, population, run_id = sample_id.split("_")
	return (
		PROJECT_ROOT
		/ "results/01_social_interaction"
		/ setting
		/ f"model_{model_id}"
		/ population
		/ f"results_{model_id}_{population}_{run_id}"
	)


DEFAULT_WELL_BEHAVED_PATH: Path | None = _sample_path_from_id(DEFAULT_WELL_BEHAVED_SAMPLE_ID)
DEFAULT_VOLATILE_PATH: Path | None = _sample_path_from_id(DEFAULT_VOLATILE_SAMPLE_ID)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/04_supplemental_figures/method"
DEFAULT_TIME_LIMIT_SECONDS = 60.0
PLOT_INTERVAL_SECONDS = 0.24
REGION_STYLES = {
	"well-behaved": {
		"color": "#4C72B0",
	},
	"volatile": {
		"color": "#C44E52",
	},
}


@dataclass(frozen=True)
class SampleDescriptor:
	model_id: str
	population: str
	run_id: str

	@property
	def sample_id(self) -> str:
		return f"{self.model_id}_{self.population}_{self.run_id}"


@dataclass(frozen=True)
class TrajectoryRecord:
	label: str
	time_seconds: list[float]
	min_eigenvalues: list[float]
	source_path: Path


def _configure_plot_style() -> None:
	sns.set_theme(style="ticks", context="paper", font_scale=1.3)
	plt.rcParams["font.family"] = "sans-serif"
	plt.rcParams["axes.linewidth"] = 1.2


def _resolve_eval_trend_path(sample_path: Path) -> Path:
	"""Resolve a user path to ``eval_trend_at_opt.json``."""
	path = sample_path.expanduser().resolve()
	if not path.exists():
		raise FileNotFoundError(f"Sample path does not exist: {path}")

	if path.is_file():
		if path.name == "eval_trend_at_opt.json":
			return path
		if path.name == "summary.json":
			candidate = path.parent / "eval_trend_at_opt.json"
			if candidate.is_file():
				return candidate
		raise FileNotFoundError(
			f"Expected eval_trend_at_opt.json or summary.json, got: {path}"
		)

	candidate = path / "eval_trend_at_opt.json"
	if candidate.is_file():
		return candidate

	raise FileNotFoundError(
		f"Could not find eval_trend_at_opt.json under sample directory: {path}"
	)


def _parse_sample_descriptor(sample_path: Path) -> SampleDescriptor:
	eval_path = _resolve_eval_trend_path(sample_path)
	results_name = eval_path.parent.name
	if not results_name.startswith("results_"):
		raise ValueError(
			f"Could not parse sample descriptor from results directory: {eval_path.parent}"
		)

	model_id, population, run_id = results_name.removeprefix("results_").split("_")
	return SampleDescriptor(
		model_id=model_id,
		population=population,
		run_id=run_id,
	)


def _comparison_output_path(
	output_dir: Path,
	well_behaved: SampleDescriptor,
	volatile: SampleDescriptor,
) -> Path:
	comparison_dir = output_dir / f"{well_behaved.model_id}_vs_{volatile.model_id}"
	filename = f"{well_behaved.sample_id}_vs_{volatile.sample_id}.pdf"
	return comparison_dir / filename


def _sample_label(sample_path: Path, eval_path: Path) -> str:
	if sample_path.is_file():
		sample_path = sample_path.parent
	parts = eval_path.parts
	for index, part in enumerate(parts):
		if part.startswith("model_") and index + 2 < len(parts):
			model_id = part
			population = parts[index + 2]
			sample_name = eval_path.parent.name
			return f"{model_id}, {population}, {sample_name}"
	return eval_path.parent.name


def load_trajectory(sample_path: Path) -> TrajectoryRecord:
	eval_path = _resolve_eval_trend_path(sample_path)
	with open(eval_path, "r") as f:
		payload = json.load(f)

	time_seconds = [float(x) for x in payload.get("time_seconds", [])]
	min_eigenvalues = [float(x) for x in payload.get("min_eigenvalues", [])]
	if not time_seconds or len(time_seconds) != len(min_eigenvalues):
		raise ValueError(
			f"Invalid trajectory data in {eval_path}: "
			f"len(time_seconds)={len(time_seconds)}, "
			f"len(min_eigenvalues)={len(min_eigenvalues)}"
		)

	return TrajectoryRecord(
		label=_sample_label(sample_path, eval_path),
		time_seconds=time_seconds,
		min_eigenvalues=min_eigenvalues,
		source_path=eval_path,
	)


def _truncate_trajectory(
	record: TrajectoryRecord,
	time_limit_seconds: float | None,
) -> TrajectoryRecord:
	if time_limit_seconds is None:
		return record

	time_seconds: list[float] = []
	min_eigenvalues: list[float] = []
	for time_value, min_eigenvalue in zip(record.time_seconds, record.min_eigenvalues):
		if time_value > time_limit_seconds:
			break
		time_seconds.append(time_value)
		min_eigenvalues.append(min_eigenvalue)

	if not time_seconds:
		raise ValueError(
			f"No trajectory points remain for {record.source_path} "
			f"with time limit {time_limit_seconds} s."
		)

	return TrajectoryRecord(
		label=record.label,
		time_seconds=time_seconds,
		min_eigenvalues=min_eigenvalues,
		source_path=record.source_path,
	)


def _sample_trajectory(
	record: TrajectoryRecord,
	interval_seconds: float,
) -> TrajectoryRecord:
	"""Select the point nearest each interval boundary."""
	if interval_seconds <= 0.0:
		raise ValueError("interval_seconds must be positive.")

	sampled_times: list[float] = []
	sampled_values: list[float] = []
	point_index = 0
	target_time = interval_seconds
	last_time = record.time_seconds[-1]

	while target_time <= last_time + 1e-9:
		while (
			point_index + 1 < len(record.time_seconds)
			and abs(record.time_seconds[point_index + 1] - target_time)
			< abs(record.time_seconds[point_index] - target_time)
		):
			point_index += 1
		sampled_times.append(record.time_seconds[point_index])
		sampled_values.append(record.min_eigenvalues[point_index])
		target_time += interval_seconds

	if not sampled_times:
		raise ValueError(
			f"No trajectory points remain for {record.source_path} "
			f"with sampling interval {interval_seconds} s."
		)

	return TrajectoryRecord(
		label=record.label,
		time_seconds=sampled_times,
		min_eigenvalues=sampled_values,
		source_path=record.source_path,
	)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
	fig.savefig(output_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {output_path}")


def plot_min_eigenvalue_trajectories(
	records: dict[str, TrajectoryRecord],
	output_path: Path,
	time_limit_seconds: float | None,
) -> None:
	fig, ax = plt.subplots(figsize=(5.5, 3.5))

	for region_key in ("well-behaved", "volatile"):
		record = records[region_key]
		style = REGION_STYLES[region_key]
		ax.plot(
			record.time_seconds,
			record.min_eigenvalues,
			color=style["color"],
			linewidth=1.8,
		)

	ax.set_xlabel("Time [s]")
	ax.set_ylabel(
		r"Minimum eigenvalue" + "\n" + r"$\lambda^{\min}(\mathcal{H}_T)$",
		linespacing=1.4,
	)
	if time_limit_seconds is not None:
		ax.set_xlim(0.0, time_limit_seconds)
	ax.grid(True, which="both", linestyle="--", alpha=0.25)
	sns.despine(ax=ax)

	fig.tight_layout()
	_save_figure(fig, output_path)


def _resolve_cli_path(path: Path | None, fallback: Path | None, option_name: str) -> Path:
	resolved = path if path is not None else fallback
	if resolved is None:
		raise ValueError(
			f"{option_name} is required. Pass it on the command line or set the "
			f"corresponding DEFAULT_* constant at the top of this script."
		)
	return resolved


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Plot minimum-eigenvalue trajectories for selected model samples."
	)
	parser.add_argument(
		"--well-behaved-path",
		type=Path,
		default=DEFAULT_WELL_BEHAVED_PATH,
		help=(
			"Path to a sample result directory, summary.json, or eval_trend_at_opt.json "
			"for the well-behaved example."
		),
	)
	parser.add_argument(
		"--volatile-path",
		type=Path,
		default=DEFAULT_VOLATILE_PATH,
		help=(
			"Path to a sample result directory, summary.json, or eval_trend_at_opt.json "
			"for the volatile example."
		),
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=DEFAULT_OUTPUT_DIR,
		help="Directory where the figure is written.",
	)
	parser.add_argument(
		"--time-limit",
		type=float,
		default=DEFAULT_TIME_LIMIT_SECONDS,
		help="Maximum time shown on the x-axis in seconds (default: 300).",
	)
	parser.add_argument(
		"--plot-to-end",
		action="store_true",
		help="Plot the full trajectory instead of truncating at --time-limit.",
	)
	args = parser.parse_args()

	_configure_plot_style()
	output_dir = args.output_dir.expanduser().resolve()
	time_limit_seconds = None if args.plot_to_end else args.time_limit
	well_behaved_path = _resolve_cli_path(
		args.well_behaved_path,
		DEFAULT_WELL_BEHAVED_PATH,
		"--well-behaved-path",
	)
	volatile_path = _resolve_cli_path(
		args.volatile_path,
		DEFAULT_VOLATILE_PATH,
		"--volatile-path",
	)

	well_behaved_descriptor = _parse_sample_descriptor(well_behaved_path)
	volatile_descriptor = _parse_sample_descriptor(volatile_path)

	well_behaved_record = _truncate_trajectory(
		load_trajectory(well_behaved_path),
		time_limit_seconds,
	)
	well_behaved_record = _sample_trajectory(
		well_behaved_record,
		PLOT_INTERVAL_SECONDS,
	)
	volatile_record = _truncate_trajectory(
		load_trajectory(volatile_path),
		time_limit_seconds,
	)
	volatile_record = _sample_trajectory(
		volatile_record,
		PLOT_INTERVAL_SECONDS,
	)

	print(
		f"Well-behaved sample: model_{well_behaved_descriptor.model_id}, "
		f"{well_behaved_descriptor.population}, "
		f"run {well_behaved_descriptor.run_id}"
	)
	print(f"  {well_behaved_record.source_path}")
	print(
		f"Volatile sample: model_{volatile_descriptor.model_id}, "
		f"{volatile_descriptor.population}, "
		f"run {volatile_descriptor.run_id}"
	)
	print(f"  {volatile_record.source_path}")
	if time_limit_seconds is None:
		print("Time range: full trajectory")
	else:
		print(f"Time range: 0–{time_limit_seconds} s")
	print(f"Plot interval: {PLOT_INTERVAL_SECONDS} s")

	output_path = _comparison_output_path(
		output_dir,
		well_behaved_descriptor,
		volatile_descriptor,
	)
	print(f"Output directory: {output_path.parent}")

	plot_min_eigenvalue_trajectories(
		{
			"well-behaved": well_behaved_record,
			"volatile": volatile_record,
		},
		output_path,
		time_limit_seconds,
	)


if __name__ == "__main__":
	main()
