from __future__ import annotations

from pathlib import Path

import toml

PRESETS_ROOT = Path(__file__).parent / "parameter_presets"


def load_preset(path: Path) -> dict:
    raw = toml.load(path)
    modules = raw["modules"]
    parameters = raw.get("parameters", {})

    preset = {
        "meta": raw.get("meta", {}),
        "modules": modules,
    }
    for category, module_names in modules.items():
        category_params = parameters.get(category, {})
        preset[category] = {
            name: dict(category_params[name]) if isinstance(category_params.get(name), dict) else {}
            for name in module_names
        }
    return preset


def load_parameter_presets(group: str | None = None) -> dict[str, dict]:
    search_root = PRESETS_ROOT / group if group is not None else PRESETS_ROOT
    if not search_root.is_dir():
        raise FileNotFoundError(
            f"Parameter preset directory not found: {search_root}"
        )

    presets: dict[str, dict] = {}
    for path in sorted(search_root.rglob("setting_*.toml")):
        if group is not None:
            key = path.stem
        else:
            key = str(path.relative_to(PRESETS_ROOT).with_suffix(""))
        presets[key] = load_preset(path)
    return presets
