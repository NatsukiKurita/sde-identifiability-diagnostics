# Traced by PackageCompiler while building sys_simulation.so.
# Runs main_all.jl end to end on short trajectories for every Kinetics x Social pair in the
# parameter presets, so package methods are compiled for the argument types the simulator uses.
# Project code itself is re-included at run time; only package specializations are baked in.

using TOML

const PROJECT_DIR = dirname(@__DIR__)
const PRESETS_DIR = joinpath(dirname(PROJECT_DIR), "experiment_configs", "parameter_presets")
const PRECOMPILE_FRAMES = 48
const PRECOMPILE_NUM_AGENTS = [2, 4]

include(joinpath(PROJECT_DIR, "main_all.jl"))

function collect_module_params()
    kinetics = Dict{String, Dict{String, Any}}()
    social = Dict{String, Dict{String, Any}}()
    for (root, _, files) in walkdir(PRESETS_DIR)
        for file in files
            (startswith(file, "setting_") && endswith(file, ".toml")) || continue
            preset = TOML.parsefile(joinpath(root, file))
            params = get(preset, "parameters", Dict{String, Any}())
            for name in preset["modules"]["Kinetics"]
                kinetics[name] = get(get(params, "Kinetics", Dict()), name, Dict{String, Any}())
            end
            for name in preset["modules"]["Social"]
                social[name] = get(get(params, "Social", Dict()), name, Dict{String, Any}())
            end
        end
    end
    return kinetics, social
end

function write_configs(setting_root::String, precision::String, kinetics, social)
    model_id = 0
    for (k_name, k_params) in kinetics, (s_name, s_params) in social
        model_id += 1
        model_code = lpad(model_id, 4, '0')
        model_dir = joinpath(setting_root, "model_$(model_code)")
        mkpath(model_dir)
        config = Dict(
            "Simulation" => Dict(
                "total_simulation_per_model" => 1,
                "num_agents" => PRECOMPILE_NUM_AGENTS,
                "num_frames" => PRECOMPILE_FRAMES,
                "precision" => precision,
                "dt" => 1 / 24,
                "safe_eps" => precision == "Float32" ? 1e-6 : 1e-14,
                "v_0" => 0.05,
            ),
            "Kinetics" => merge(Dict{String, Any}("type" => k_name), k_params),
            "Social" => merge(Dict{String, Any}("type" => s_name), s_params),
        )
        open(joinpath(model_dir, "config_$(model_code).toml"), "w") do io
            TOML.print(io, config)
        end
    end
end

let
    kinetics, social = collect_module_params()
    isempty(kinetics) && error("No parameter presets found under $PRESETS_DIR")
    for precision in ("Float32", "Float64")
        setting_root = mktempdir()
        write_configs(setting_root, precision, kinetics, social)
        args = parse_commandline(["--simulation_root", setting_root, "--overwrite", "true"])
        process_all_directories(args["simulation_root"], parse_overwrite(args["overwrite"]))
    end
end
