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
)

# Adjust these figure sizes independently.
IDENTIFIABLE_FIGSIZE = (6.4, 4.0)
UNIDENTIFIABLE_FIGSIZE = (6.4, 4.0)


def plot_for_setting(setting_dir: Path, records: list[PaperRecord], _args) -> None:
	filtered = [record for record in records if record.panel == MAIN_TEXT_NUM_AGENT]
	if not filtered:
		print(f"[{setting_dir.name}] No N={MAIN_TEXT_NUM_AGENT} data for spaghetti plots; skipping.")
		return

	kinetics_order = ordered_kinetics_from_set({record.kinetics for record in filtered})
	palette = model_palette(ALL_SOCIAL_MODELS)
	output_dir = setting_dir / "figures" / "min_eigenvalue_spaghetti_by_model"
	for group in ALL_SOCIAL_MODELS:
		figsize = UNIDENTIFIABLE_FIGSIZE if group in MIN_EIGENVALUE_BOTTOM_MODELS else IDENTIFIABLE_FIGSIZE
		for kinetics in kinetics_order:
			cell = [
				record
				for record in filtered
				if record.group == group and record.kinetics == kinetics
			]
			if not cell:
				print(f"[{setting_dir.name}] No {group} {kinetics} samples; skipping.")
				continue
			plot_spaghetti_min_eigenvalue(
				cell,
				output_dir / f"{group}_{kinetics}.pdf",
				figsize=figsize,
				color=palette[group],
				title=f"{group} ({kinetics})",
			)


if __name__ == "__main__":
	run_for_settings(
		plot_for_setting,
		description="Spaghetti min-eigenvalue charts for every social model.",
		apply_exclusions=False,
	)
