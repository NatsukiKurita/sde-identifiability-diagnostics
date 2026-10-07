#!/usr/bin/env python3
"""Example of the KR2021 (Inertial, N=2) tandem state.

The KR2021 positional force acts only through the component perpendicular to the
heading. When one individual swims directly behind the other, that component
vanishes and the distance is not restored to ``d_eq``; the pair keeps whatever
distance it had when it entered the tandem. This script plots 10 seconds right
after burn-in in the reference individual's body frame (heading along +x), for a
tandem sample and a typical side-by-side sample. Outputs are written under
``results/04_supplemental_figures/kr2021_tandem``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "04_figures"))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import toml

from _common import DASHED_LINESTYLE, configure_plot_style, model_palette, save_paper_figure
from src.experiment_configs.master_constants import MasterConstants
from src.paths import DATA_ROOT, PROJECT_ROOT

SETTING = "setting_1"
MODEL = "KR2021"
MODEL_ID = "0006"
NUM_AGENTS = 2
TANDEM_SAMPLE_ID = "0041"
SECOND_TANDEM_SAMPLE_ID = "0038"
TYPICAL_SAMPLE_ID = "0001"
BURN_IN_SECONDS = float(MasterConstants.BURN_IN_SECONDS)
DURATION_SECONDS = 10.0
OUTPUT_PATH = PROJECT_ROOT / "results/04_supplemental_figures/kr2021_tandem/kr2021_tandem_body_frame.pdf"
FIGSIZE = (5.2, 5.2)
AXIS_LIMIT = 2.0


def _partner_in_body_frame(h5_path: Path, dt: float, d_eq: float, *, use_last_window: bool = False) -> np.ndarray:
	n_window = int(round(DURATION_SECONDS / dt)) + 1
	with h5py.File(h5_path, "r") as f:
		n_total = f["positions"].shape[0]
		start = n_total - n_window if use_last_window else int(round(BURN_IN_SECONDS / dt))
		stop = start + n_window
		positions = np.asarray(f["positions"][start:stop], dtype=float)
		orientations = np.asarray(f["orientations"][start:stop], dtype=float)

	relative = positions[:, 1] - positions[:, 0]
	heading_0 = np.stack([np.cos(orientations[:, 0]), np.sin(orientations[:, 0])], axis=-1)
	# Use the leader as reference so that the partner appears behind it (x < 0) in a tandem.
	reference = 0 if np.mean(np.sum(relative * heading_0, axis=-1)) < 0.0 else 1
	partner = 1 - reference
	relative = positions[:, partner] - positions[:, reference]
	theta = orientations[:, reference]
	forward = relative[:, 0] * np.cos(theta) + relative[:, 1] * np.sin(theta)
	lateral = -relative[:, 0] * np.sin(theta) + relative[:, 1] * np.cos(theta)
	return np.stack([forward, lateral], axis=-1) / d_eq


def main() -> None:
	configure_plot_style()
	model_dir = DATA_ROOT / "simulation" / SETTING / f"model_{MODEL_ID}"
	config = toml.load(model_dir / f"config_{MODEL_ID}.toml")
	dt = float(config["Simulation"]["dt"])
	d_eq = float(config["Social"]["d_eq"])
	sample_dir = model_dir / "ground_truth" / f"N{NUM_AGENTS}"

	samples = (
		(TYPICAL_SAMPLE_ID, "Typical", "0.6", "o", False),
		(TANDEM_SAMPLE_ID, "Tandem", model_palette([MODEL])[MODEL], "o", False),
		(SECOND_TANDEM_SAMPLE_ID, "Tandem, last 10 s", model_palette(["TK2020"])["TK2020"], "s", True),
	)

	fig, ax = plt.subplots(figsize=FIGSIZE)
	ring = np.linspace(0.0, 2.0 * np.pi, 361)
	ax.plot(np.cos(ring), np.sin(ring), color="0.25", linestyle=DASHED_LINESTYLE, linewidth=1.1, zorder=1)
	ax.text(np.cos(np.pi / 4), np.sin(np.pi / 4), r"  $d = d^{\mathrm{eq}}$", color="0.2", fontsize=9, ha="left", va="bottom")

	for sample_id, label, color, marker, use_last_window in samples:
		h5_path = sample_dir / f"simulation_{MODEL_ID}_N{NUM_AGENTS}_{sample_id}.h5"
		partner = _partner_in_body_frame(h5_path, dt, d_eq, use_last_window=use_last_window)
		ax.plot(
			partner[:, 0],
			partner[:, 1],
			color=color,
			linewidth=0.8,
			marker=marker,
			markersize=2.5,
			alpha=0.8,
			label=f"{label} (sample {sample_id})",
			zorder=3,
		)

	ax.scatter(0.0, 0.0, color="0.15", s=30, zorder=4)
	ax.annotate(
		"",
		xy=(0.6, 0.0),
		xytext=(0.0, 0.0),
		arrowprops={"arrowstyle": "-|>", "color": "0.15", "lw": 1.3, "mutation_scale": 12},
		zorder=4,
	)
	ax.text(0.0, -0.12, "Reference", color="0.15", fontsize=9, ha="center", va="top")

	ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
	ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel(r"Forward position of partner / $d^{\mathrm{eq}}$")
	ax.set_ylabel(r"Lateral position of partner / $d^{\mathrm{eq}}$")
	ax.grid(True, linestyle=DASHED_LINESTYLE, alpha=0.25)
	ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=9)
	sns.despine(ax=ax)
	fig.tight_layout()
	save_paper_figure(fig, OUTPUT_PATH)


if __name__ == "__main__":
	main()
