#!/usr/bin/env python3
"""Fig 5: frequency density of the post-burn-in normalized distance d/d_eq for KR2021 (N=2)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from _common import (
	DASHED_LINESTYLE,
	KINETICS_LABELS,
	KINETICS_PLOT_LINESTYLES,
	MAIN_TEXT_NUM_AGENT,
	_load_ledger_inverse,
	_load_truth_params,
	nested_param_value,
	PaperRecord,
	run_for_settings,
	save_paper_figure,
)
from src.experiment_configs.eigenvalue_time_series import EigenvalueTimeSeriesConfig
from src.paths import DATA_ROOT

FIGSIZE = (6.4, 4.0)
MODEL = "KR2021"
X_MAX = 3.5
NUM_BINS = 350
BURN_IN_FRAMES = int(EigenvalueTimeSeriesConfig.BURN_IN)


def _model_ids_by_kinetics(setting_name: str, model: str) -> dict[str, str]:
	model_ids: dict[str, str] = {}
	for model_num, combo_key in _load_ledger_inverse(setting_name).items():
		kinetics, social = (part.split("_", 1)[1] for part in combo_key.split("-", 1))
		if social == model:
			model_ids[kinetics] = f"{model_num:04d}"
	return model_ids


def _post_burn_in_distances(setting_name: str, model_id: str, num_agents: int) -> np.ndarray:
	sample_dir = DATA_ROOT / "simulation" / setting_name / f"model_{model_id}" / "ground_truth" / f"N{num_agents}"
	distances: list[np.ndarray] = []
	for h5_path in sorted(sample_dir.glob(f"simulation_{model_id}_N{num_agents}_*.h5")):
		with h5py.File(h5_path, "r") as f:
			positions = np.asarray(f["positions"][BURN_IN_FRAMES:], dtype=float)
		distances.append(np.linalg.norm(positions[:, 0, :] - positions[:, 1, :], axis=-1))
	if not distances:
		return np.empty(0)
	print(f"[{setting_name}] model_{model_id}: {len(distances)} samples after {BURN_IN_FRAMES} burn-in frames")
	return np.concatenate(distances)


def plot_for_setting(setting_dir: Path, _records: list[PaperRecord], _args) -> None:
	setting_name = setting_dir.name
	model_ids = _model_ids_by_kinetics(setting_name, MODEL)
	distances_by_kinetics = {
		kinetics: _post_burn_in_distances(setting_name, model_ids[kinetics], MAIN_TEXT_NUM_AGENT)
		for kinetics in ("Overdamped", "Inertial")
		if kinetics in model_ids
	}
	distances_by_kinetics = {kinetics: d for kinetics, d in distances_by_kinetics.items() if d.size}
	if not distances_by_kinetics:
		print(f"[{setting_name}] No {MODEL} N={MAIN_TEXT_NUM_AGENT} trajectories; skipping.")
		return

	truth_params = _load_truth_params(setting_name, next(iter(model_ids.values())))
	d_eq = nested_param_value(truth_params, "Social.d_eq") if truth_params else None
	if d_eq is None or d_eq <= 0.0:
		print(f"[{setting_name}] Missing positive {MODEL} d_eq; skipping.")
		return
	ratios_by_kinetics = {kinetics: d / d_eq for kinetics, d in distances_by_kinetics.items()}

	bins = np.linspace(0.0, X_MAX, NUM_BINS + 1)
	palette = dict(zip(ratios_by_kinetics, sns.color_palette("deep", n_colors=len(ratios_by_kinetics))))

	fig, ax = plt.subplots(figsize=FIGSIZE)
	for kinetics, ratios in ratios_by_kinetics.items():
		color = palette[kinetics]
		ax.hist(ratios, bins=bins, density=True, color=color, alpha=0.25, linewidth=0)
		ax.hist(
			ratios,
			bins=bins,
			density=True,
			histtype="step",
			color=color,
			linestyle=KINETICS_PLOT_LINESTYLES.get(kinetics, "-"),
			linewidth=1.5,
			label=KINETICS_LABELS.get(kinetics, kinetics),
		)

	ax.axvline(1.0, color="0.35", linestyle="-", linewidth=0.9, zorder=0)
	ax.set_xlim(0.0, X_MAX)
	ax.set_xlabel(r"$d / d^{\mathrm{eq}}$")
	ax.set_ylabel("Relative frequency")
	ax.grid(True, linestyle=DASHED_LINESTYLE, alpha=0.25)
	ax.legend(frameon=False, loc="upper right")
	sns.despine(ax=ax)
	fig.tight_layout()
	save_paper_figure(fig, setting_dir / "figures" / "relative_distance" / f"{MODEL}.pdf")


if __name__ == "__main__":
	run_for_settings(
		plot_for_setting,
		description="Relative frequency of KR2021 inter-individual distance after burn-in.",
	)
