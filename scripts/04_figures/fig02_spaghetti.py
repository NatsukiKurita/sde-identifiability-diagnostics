#!/usr/bin/env python3
"""Fig 2: spaghetti min-eigenvalue charts for every social model and kinetics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
	ALL_SOCIAL_MODELS,
	MAIN_TEXT_NUM_AGENT,
	MIN_EIGENVALUE_BOTTOM_MODELS,
	PaperRecord,
	model_palette,
	ordered_kinetics_from_set,
	plot_spaghetti_min_eigenvalue,
	run_for_settings,
	split_kr2021_regimes,
)

# Adjust these figure sizes independently.
IDENTIFIABLE_FIGSIZE = (6.4, 4.0)
UNIDENTIFIABLE_FIGSIZE = (6.4, 4.0)
# (main-axes guide, inset) per KR2021 regime; the linear regime grows linearly near the end.
KR2021_REGIME_GUIDE_INSETS = {
	"linear": (None, "slope_one_tail"),
	"fluctuation": ("sqrt_log", None),
}
# Merged KR2021 Inertial panel: both regimes on one axes.
KR2021_INERTIAL_GUIDE_INSET = ("sqrt_log", "slope_one_tail")


def _spaghetti_panels(
	group: str,
	kinetics: str,
	cell: list[PaperRecord],
	*,
	num_agents: int,
	kr2021_split_by_estimate: bool,
	tk2020_linear_tail_inset: bool,
) -> list[tuple[list[PaperRecord], str, str, str | None, str | None]]:
	if group == "KR2021" and not kr2021_split_by_estimate:
		return [
			(
				cell,
				f"{group}_{kinetics}",
				f"{group} ({kinetics})",
				*KR2021_REGIME_GUIDE_INSETS["linear"],
			)
		]
	if group == "KR2021" and kinetics == "Inertial":
		return [
			(
				cell,
				f"{group}_{kinetics}",
				f"{group} ({kinetics})",
				*KR2021_INERTIAL_GUIDE_INSET,
			)
		]
	if group == "KR2021":
		return [
			(
				regime_records,
				f"{group}_{kinetics}_{regime}",
				f"{group} ({kinetics}, {regime})",
				*KR2021_REGIME_GUIDE_INSETS[regime],
			)
			for regime, regime_records in split_kr2021_regimes(cell).items()
		]
	if group == "TK2020" and num_agents == MAIN_TEXT_NUM_AGENT:
		inset = "linear_tail" if tk2020_linear_tail_inset else None
		return [(cell, f"{group}_{kinetics}", f"{group} ({kinetics})", "sqrt_log", inset)]
	if group == "TK2020":
		return [(cell, f"{group}_{kinetics}", f"{group} ({kinetics}, N={num_agents})", "slope_one", None)]
	return [(cell, f"{group}_{kinetics}", f"{group} ({kinetics})", "slope_one", None)]


def plot_for_setting(
	setting_dir: Path,
	records: list[PaperRecord],
	_args,
	*,
	num_agents: int = MAIN_TEXT_NUM_AGENT,
	figures_dir: Path | None = None,
	kr2021_split_by_estimate: bool = True,
	tk2020_linear_tail_inset: bool = True,
) -> None:
	filtered = [record for record in records if record.panel == num_agents]
	if not filtered:
		print(f"[{setting_dir.name}] No N={num_agents} data for spaghetti plots; skipping.")
		return

	kinetics_order = ordered_kinetics_from_set({record.kinetics for record in filtered})
	palette = model_palette(ALL_SOCIAL_MODELS)
	output_dir = (figures_dir or setting_dir / "figures") / "spaghetti"
	for group in ALL_SOCIAL_MODELS:
		figsize = UNIDENTIFIABLE_FIGSIZE if group in MIN_EIGENVALUE_BOTTOM_MODELS else IDENTIFIABLE_FIGSIZE
		for kinetics in kinetics_order:
			cell = [
				record
				for record in filtered
				if record.group == group and record.kinetics == kinetics
			]
			if not cell:
				print(f"[{setting_dir.name}] No {group} {kinetics} N={num_agents} samples; skipping.")
				continue
			for panel_records, stem, title, guide, inset in _spaghetti_panels(
				group,
				kinetics,
				cell,
				num_agents=num_agents,
				kr2021_split_by_estimate=kr2021_split_by_estimate,
				tk2020_linear_tail_inset=tk2020_linear_tail_inset,
			):
				if not panel_records:
					print(f"[{setting_dir.name}] No samples for {stem}; skipping.")
					continue
				plot_spaghetti_min_eigenvalue(
					panel_records,
					output_dir / f"{stem}.pdf",
					figsize=figsize,
					color=palette[group],
					guide=guide,
					inset=inset,
					title=title,
				)


if __name__ == "__main__":
	run_for_settings(
		plot_for_setting,
		description="Spaghetti min-eigenvalue charts for every social model.",
	)
