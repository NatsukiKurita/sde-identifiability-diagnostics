#!/usr/bin/env python3
"""Plot example fish trajectories for every simulation model.

For each setting under ``data/simulation`` and each model/configuration, this
script plots sample ``0001`` for 15 seconds after burn-in. Outputs are written under
``results/04_supplemental_figures/method/example_trajectories`` while preserving
the relative directory structure below ``data/simulation``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiment_configs.master_constants import MasterConstants
from src.paths import DATA_ROOT, PROJECT_ROOT

DEFAULT_SIMULATION_ROOT = DATA_ROOT / "simulation"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/04_supplemental_figures/method/example_trajectories"
DEFAULT_SAMPLE_ID = "0001"
DEFAULT_DURATION_SECONDS = 15.0
DEFAULT_BURN_IN_SECONDS = float(MasterConstants.BURN_IN_SECONDS)


@dataclass(frozen=True)
class TrajectoryJob:
	config_path: Path
	h5_path: Path
	output_path: Path
	setting: str
	model_id: str
	population: str
	sample_id: str
	dt: float
	duration_seconds: float
	burn_in_seconds: float


def _configure_plot_style() -> None:
	sns.set_theme(style="ticks", context="paper", font_scale=1.2)
	plt.rcParams["font.family"] = "sans-serif"
	plt.rcParams["axes.linewidth"] = 1.2


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, format="pdf", dpi=300, bbox_inches="tight")
	fig.savefig(output_path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {output_path}")


def _output_exists(output_path: Path) -> bool:
	return output_path.exists() or output_path.with_suffix(".png").exists()


def _normalize_positions(positions: np.ndarray) -> np.ndarray:
	if positions.ndim != 3:
		raise ValueError(f"positions must be 3D, got shape {positions.shape}")
	if positions.shape[-1] == 2:
		return positions
	if positions.shape[0] == 2:
		return np.transpose(positions, (2, 1, 0))
	raise ValueError(f"Cannot infer coordinate axis for positions with shape {positions.shape}")


def _model_label_from_ledger(setting_dir: Path, model_id: str) -> str:
	ledger_path = setting_dir / "ledger.json"
	if not ledger_path.is_file():
		return f"model_{model_id}"

	with open(ledger_path, "r") as f:
		ledger = json.load(f)
	for combo_key, ledger_model_id in ledger.items():
		if int(ledger_model_id) == int(model_id):
			return combo_key
	return f"model_{model_id}"


def _short_model_label(combo_key: str) -> str:
	parts = combo_key.split("-")
	if len(parts) >= 2:
		kinetics = parts[0]
		social = parts[1].replace("Soc_", "")
		return f"{kinetics} / {social}"
	return combo_key


def _parse_model_id(config_path: Path) -> str:
	return config_path.parent.name.removeprefix("model_")


def _simulation_path(
	config_path: Path,
	model_id: str,
	population: int,
	sample_id: str,
) -> Path:
	return (
		config_path.parent
		/ "ground_truth"
		/ f"N{population}"
		/ f"simulation_{model_id}_N{population}_{sample_id}.h5"
	)


def _output_path_for_h5(
	h5_path: Path,
	simulation_root: Path,
	output_root: Path,
) -> Path:
	relative_path = h5_path.relative_to(simulation_root)
	return output_root / relative_path.with_suffix(".pdf")


def discover_jobs(
	simulation_root: Path,
	output_root: Path,
	sample_id: str,
	duration_seconds: float,
	burn_in_seconds: float,
) -> list[TrajectoryJob]:
	jobs: list[TrajectoryJob] = []

	for config_path in sorted(simulation_root.glob("setting_*/model_*/config_*.toml")):
		config_toml = toml.load(config_path)
		model_id = _parse_model_id(config_path)
		dt = float(config_toml["Simulation"]["dt"])
		setting = config_path.relative_to(simulation_root).parts[0]

		for population in config_toml["Simulation"]["num_agents"]:
			h5_path = _simulation_path(config_path, model_id, int(population), sample_id)
			if not h5_path.is_file():
				print(f"Warning: missing sample trajectory, skipping: {h5_path}")
				continue

			jobs.append(
				TrajectoryJob(
					config_path=config_path,
					h5_path=h5_path,
					output_path=_output_path_for_h5(h5_path, simulation_root, output_root),
					setting=setting,
					model_id=model_id,
					population=f"N{int(population)}",
					sample_id=sample_id,
					dt=dt,
					duration_seconds=duration_seconds,
					burn_in_seconds=burn_in_seconds,
				)
			)

	return jobs


def plot_example_trajectory(job: TrajectoryJob) -> None:
	with h5py.File(job.h5_path, "r") as f:
		positions = _normalize_positions(np.asarray(f["positions"][:], dtype=float))

	start_frame = int(round(job.burn_in_seconds / job.dt))
	frame_count = int(round(job.duration_seconds / job.dt)) + 1
	end_frame = min(positions.shape[0], start_frame + frame_count)
	if end_frame - start_frame < 2:
		raise ValueError(f"Not enough frames in {job.h5_path}")

	positions = positions[start_frame:end_frame]
	time_start = start_frame * job.dt
	time_end = (end_frame - 1) * job.dt
	plot_duration = time_end - time_start
	model_label = _short_model_label(_model_label_from_ledger(job.config_path.parents[1], job.model_id))

	fig, ax = plt.subplots(figsize=(5.0, 5.0))

	palette = sns.color_palette("husl", n_colors=positions.shape[1])
	for agent_idx in range(positions.shape[1]):
		agent_positions = positions[:, agent_idx, :]
		color = palette[agent_idx]
		ax.plot(agent_positions[:, 0], agent_positions[:, 1], color=color, linewidth=1.0, alpha=0.8)
		ax.scatter(agent_positions[0, 0], agent_positions[0, 1], color=color, s=20, marker="o", zorder=3)
		ax.scatter(agent_positions[-1, 0], agent_positions[-1, 1], color=color, s=24, marker="x", zorder=3)

	ax.set_title(
		f"{job.setting} model_{job.model_id}\n"
		f"{model_label}, {job.population}, "
		f"{time_start:.1f}-{time_end:.1f} s ({plot_duration:.1f} s)"
	)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_aspect("equal", adjustable="box")
	ax.grid(True, linestyle="--", alpha=0.2)
	sns.despine(ax=ax)
	fig.tight_layout()
	_save_figure(fig, job.output_path)


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot example trajectories for every simulation model.")
	parser.add_argument(
		"--simulation-root",
		type=Path,
		default=DEFAULT_SIMULATION_ROOT,
		help="Root directory containing setting/model simulation data.",
	)
	parser.add_argument(
		"--output-root",
		type=Path,
		default=DEFAULT_OUTPUT_ROOT,
		help="Output root for example trajectory figures.",
	)
	parser.add_argument(
		"--sample-id",
		type=str,
		default=DEFAULT_SAMPLE_ID,
		help="Four-digit simulation sample ID to plot.",
	)
	parser.add_argument(
		"--duration",
		type=float,
		default=DEFAULT_DURATION_SECONDS,
		help="Trajectory duration in seconds after burn-in.",
	)
	parser.add_argument(
		"--burn-in",
		type=float,
		default=DEFAULT_BURN_IN_SECONDS,
		help="Burn-in duration in seconds before trajectory plotting starts.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing PDF/PNG outputs instead of skipping them.",
	)
	args = parser.parse_args()

	_configure_plot_style()
	simulation_root = args.simulation_root.expanduser().resolve()
	output_root = args.output_root.expanduser().resolve()
	jobs = discover_jobs(
		simulation_root=simulation_root,
		output_root=output_root,
		sample_id=args.sample_id,
		duration_seconds=float(args.duration),
		burn_in_seconds=float(args.burn_in),
	)
	if not jobs:
		raise FileNotFoundError(f"No trajectory jobs found under {simulation_root}")

	print(f"Discovered {len(jobs)} trajectory jobs.")
	skipped = 0
	for job in jobs:
		if not args.overwrite and _output_exists(job.output_path):
			skipped += 1
			print(f"Skipping existing output: {job.output_path}")
			continue
		plot_example_trajectory(job)
	if skipped:
		print(f"Skipped {skipped} existing outputs. Use --overwrite to regenerate them.")


if __name__ == "__main__":
	main()
