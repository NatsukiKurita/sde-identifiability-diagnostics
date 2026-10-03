#!/usr/bin/env python3
"""Run social-interaction simulations.

Examples:
    python scripts/01_simulation/01_run_simulation.py
    python scripts/01_simulation/01_run_simulation.py --setting setting_1 --overwrite
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
	jl_root = PROJECT_ROOT / "src/QMLE/estimator_modules/Neighbor/jl_delaunay_restoration"
	jl_project_path = jl_root / "project_config"
	jl_script_path = jl_root / "delaunay_restoration_backend.jl"

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
			str(jl_script_path),
			"--data_h5_root",
			str(data_h5_root),
		],
		check=True,
	)


def resolve_sysimage(use_sysimage: bool) -> Path | None:
	from src.paths import JULIA_PROJECT_DIR, JULIA_SYSIMAGE_PATH

	if not use_sysimage:
		print("Sysimage disabled (--no-sysimage); Julia will compile packages at startup.")
		return None
	if not JULIA_SYSIMAGE_PATH.is_file():
		print(
			f"Sysimage not found at {JULIA_SYSIMAGE_PATH}; running without it. "
			"Build it with ./run_setup.sh julia for faster simulations."
		)
		return None
	# A sysimage built against an older Manifest bundles stale package versions.
	manifest_path = JULIA_PROJECT_DIR / "Manifest.toml"
	if manifest_path.is_file() and manifest_path.stat().st_mtime > JULIA_SYSIMAGE_PATH.stat().st_mtime:
		print(
			f"Warning: {manifest_path.name} is newer than the sysimage. "
			"Rebuild it with ./run_setup.sh julia if package versions changed."
		)
	print(f"Using sysimage: {JULIA_SYSIMAGE_PATH}")
	return JULIA_SYSIMAGE_PATH


def run_simulation_for_setting(setting_root: Path, overwrite: bool, sysimage: Path | None) -> None:
	from src.paths import JULIA_PROJECT_DIR

	julia_script_path = JULIA_PROJECT_DIR / "main_all.jl"

	command = ["julia", f"--project={JULIA_PROJECT_DIR}"]
	if sysimage is not None:
		command.append(f"--sysimage={sysimage}")
	command += [
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
	parser.add_argument(
		"--no-sysimage",
		action="store_true",
		help="Do not load the prebuilt Julia sysimage even if it exists.",
	)
	args = parser.parse_args()

	from src.experiment_configs.simulation import SocialSimConfig
	from src.paths import DATA_ROOT

	config = SocialSimConfig
	base = Path(args.simulation_root) if args.simulation_root else DATA_ROOT / "simulation"
	setting_roots = resolve_setting_roots(config, base, args.setting)
	sysimage = resolve_sysimage(use_sysimage=not args.no_sysimage)

	for setting_root in setting_roots:
		if not setting_root.is_dir():
			print(f"Warning: {setting_root} does not exist. Skipping.")
			continue
		print(f"Running simulations in {setting_root}")
		run_simulation_for_setting(setting_root, args.overwrite, sysimage)

	if not args.skip_restore_delaunay:
		restore_delaunay_neighbors_all(base)


if __name__ == "__main__":
	main()
