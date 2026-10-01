from __future__ import annotations

from pathlib import Path

import toml

PRESETS_ROOT = Path(__file__).parent / "parameter_presets"
MODULE_CATEGORIES = ("Kinetics", "Social", "WallAvoidance", "Neighbor")


def _resolve_module_params(
    parameters: dict,
    category: str,
    module_file: str,
) -> dict:
    category_params = parameters.get(category, {})
    if module_file in category_params:
        value = category_params[module_file]
        return dict(value) if isinstance(value, dict) else {}

    stem = module_file.removesuffix(".jl")
    if stem not in category_params:
        return {}

    entry = category_params[stem]
    if not isinstance(entry, dict):
        return {}

    # TOML parses `Foo.jl` table headers as nested `{ "jl": {...} }`.
    if set(entry.keys()) == {"jl"}:
        inner = entry["jl"]
        return dict(inner) if isinstance(inner, dict) else {}

    return dict(entry)


def load_preset(path: Path) -> dict:
    raw = toml.load(path)
    modules = raw["modules"]
    parameters = raw.get("parameters", {})

    preset = {
        "meta": raw.get("meta", {}),
        "modules": modules,
    }
    for category in MODULE_CATEGORIES:
        module_files = modules[category]
        preset[category] = {
            module_file: _resolve_module_params(
                parameters,
                category,
                module_file,
            )
            for module_file in module_files
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
