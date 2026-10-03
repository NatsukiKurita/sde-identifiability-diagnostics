#!/usr/bin/env python3
"""Run social-interaction simulations.

Examples:
    python scripts/01_simulation/01_run_simulation.py
    python scripts/01_simulation/01_run_simulation.py --setting setting_01_1 --overwrite
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def restore_delaunay_neighbors_all(data_h5_root: Path) -> None:
	jl_project_path = "src/parameter_estimation/estimator_modules/Neighbor/jl_delaunay_restoration/project_config"
	jl_script_path = "src/parameter_estimation/estimator_modules/Neighbor/jl_delaunay_restoration/delaunay_restoration_backend.jl"

	subprocess.run(
		[
			"julia",
			f"--project={jl_project_path}",
			"-e",
			"using Pkg; Pkg.instantiate()",
		],
		check=True,
	)

	subprocess.run(
		[
			"julia",
			f"--project={jl_project_path}",
			jl_script_path,
			"--data_h5_root",
			str(data_h5_root),
		],
		check=True,
	)


def run_simulation_for_setting(setting_root: Path, overwrite: bool) -> None:
	julia_project_dir = PROJECT_ROOT / "src/dyn_models"
	julia_script_path = julia_project_dir / "main_all.jl"

	command = [
		"julia",
		f"--project={julia_project_dir}",
		str(julia_script_path),
		"--simulation_root",
		str(setting_root),
		"--overwrite",
		str(overwrite),
	]
	subprocess.run(command, check=True)


def resolve_setting_roots(config, base: Path, setting: str | None) -> list[Path]:
	if setting is not None:
		return [base / setting]
	return [base / setting_name for setting_name in config.PARAMETER_SETTINGS]


def main() -> None:
	parser = argparse.ArgumentParser(description="Run social-interaction simulations.")
	parser.add_argument(
		"--simulation_root",
		help="Base path containing setting_* directories. Defaults to data/simulation.",
	)
	parser.add_argument(
		"--setting",
		help="Run only this setting. Default: all settings in PARAMETER_SETTINGS.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing output files.",
	)
	parser.add_argument(
		"--skip-restore-delaunay",
		action="store_true",
		help="Skip the Delaunay-neighbor restoration step after simulation.",
	)
	args = parser.parse_args()

	from src.constants.simulation_01 import SocialSimConfig
	from src.paths import DATA_ROOT

	config = SocialSimConfig
	base = Path(args.simulation_root) if args.simulation_root else DATA_ROOT / "simulation"
	setting_roots = resolve_setting_roots(config, base, args.setting)

	for setting_root in setting_roots:
		if not setting_root.is_dir():
			print(f"Warning: {setting_root} does not exist. Skipping.")
			continue
		print(f"Running simulations in {setting_root}")
		run_simulation_for_setting(setting_root, args.overwrite)

	if not args.skip_restore_delaunay:
		restore_delaunay_neighbors_all(base)


if __name__ == "__main__":
	main()
