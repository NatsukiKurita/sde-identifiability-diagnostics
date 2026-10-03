#!/usr/bin/env python3
"""Create simulation directories and TOML configs for the social-interaction study.

Examples:
    python scripts/01_simulation/00_prepare_toml.py
    python scripts/01_simulation/00_prepare_toml.py --overwrite
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))


def _component_toml_dict(module_name: str, params: dict) -> dict:
	return {"type": module_name, **params}


def create_dir_and_toml(overwrite: bool = True) -> None:
	import toml
	from src.constants.simulation_01 import SocialSimConfig
	from src.paths import DATA_ROOT

	config = SocialSimConfig
	simdata_root = DATA_ROOT / "simulation" / config.SIMULATION_CAMPAIGN
	simdata_root.mkdir(parents=True, exist_ok=True)

	for setting_name, parameter_settings in config.PARAMETER_SETTINGS.items():
		setting_root = simdata_root / setting_name
		setting_root.mkdir(parents=True, exist_ok=True)

		ledger_path = setting_root / "ledger.json"
		if not ledger_path.exists():
			with open(ledger_path, "w") as f:
				json.dump({}, f, indent=4)
			print(f"Created ledger at {ledger_path}")

		with open(ledger_path, "r") as f:
			ledger = json.load(f)

		modules = parameter_settings["modules"]
		kinetics = modules["Kinetics"]
		social = modules["Social"]

		for k, s in itertools.product(kinetics, social):
			combo_key = f"{k}-{s}"
			if combo_key not in ledger:
				ledger[combo_key] = len(ledger) + 1

			model_id = ledger[combo_key]
			model_dir = setting_root / f"model_{model_id:04d}"
			model_dir.mkdir(parents=True, exist_ok=True)

			model_code = f"{model_id:04d}"
			toml_path = model_dir / f"config_{model_code}.toml"
			if not overwrite and toml_path.exists():
				print(f"Toml file already exists at {toml_path} and overwrite is False. Skipping.")
				continue

			dict_for_toml = {
				"Simulation": {
					"total_simulation_per_model": config.TRIALS_PER_MODEL,
					"num_agents": config.NUM_AGENTS,
					"num_frames": config.TOTAL_FRAMES,
					"precision": config.PRECISION,
					"dt": config.TIME_STEP,
					"safe_eps": config.SAFE_EPS,
					"v_0": config.ABSOLUTE_VELOCITY,
				},
				"Kinetics": _component_toml_dict(k, parameter_settings["Kinetics"][k]),
				"Social": _component_toml_dict(s, parameter_settings["Social"][s]),
			}

			with open(toml_path, "w") as f:
				toml.dump(dict_for_toml, f)

		with open(ledger_path, "w") as f:
			json.dump(ledger, f, indent=4)


def main() -> None:
	parser = argparse.ArgumentParser(description="Prepare TOML configs for social-interaction simulations.")
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing TOML files.",
	)
	args = parser.parse_args()

	create_dir_and_toml(overwrite=args.overwrite)


if __name__ == "__main__":
	main()
