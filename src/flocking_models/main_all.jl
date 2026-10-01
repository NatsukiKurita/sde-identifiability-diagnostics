include("./jl_src/SimRunner.jl")
using TOML
using ArgParse
using .SimRunner: execute_simulation

function parse_commandline()
        s = ArgParseSettings(description="Fish Schooling Simulator (TOML-driven)")
        @add_arg_table! s begin
            "--simulation_root"
                help = "path to the simulation root directory"
                arg_type = String
                required = true
            "--overwrite"
                help = "whether to overwrite existing output files (default: false)"
                arg_type = String
                default = "false"
        end
        return parse_args(s)
    end

function process_all_directories(
    simulation_base_dir::String,
    overwrite::Bool = false,
    )
    model_dirs = filter(isdir, readdir(simulation_base_dir, join=true))

    println("Julia: $(length(model_dirs)) directories found. Processing...")

    for model_dir in model_dirs
        if !startswith(basename(model_dir), "model_")
            continue
        end

        model_code = split(basename(model_dir), "_")[2]
        config_paths = filter(path -> startswith(basename(path), "config_$(model_code)"), readdir(model_dir, join=true))

        if isempty(config_paths)
            println("Warning: No TOML files found for directory: $model_dir. Skipping.")
            continue
        end

        for toml_path in config_paths
            if !isfile(toml_path)
                continue
            end

            println("Processing config: $toml_path")
            config = TOML.parsefile(toml_path)
            total_simulation_per_model = config["Simulation"]["total_simulation_per_model"]
            num_agents = config["Simulation"]["num_agents"]

            for agent_num in num_agents
                output_dir = joinpath(model_dir, "ground_truth", "N$(agent_num)")
                if !isdir(output_dir)
                    mkpath(output_dir)
                end

                for sim_num in 1:total_simulation_per_model
                    output_path = joinpath(output_dir, "simulation_$(model_code)_N$(agent_num)_$(lpad(sim_num, 4, '0')).h5")
                    if isfile(output_path) && !overwrite
                        println("Note: Output file already exists and --overwrite is set to false. Skipping simulation: $output_path")
                        continue
                    end
                    println("Simulating for agent $agent_num in directory: $model_dir")
                    execute_simulation(toml_path, output_path, agent_num, sim_num, overwrite)
                end
            end
        end
    end
    println("Julia: All simulations completed.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = parse_commandline()
    overwrite = args["overwrite"]
    if overwrite in ["True", "true", "T"]
        overwrite = true
    elseif overwrite in ["False", "false", "F"]
        overwrite = false
    else
        error("Invalid value for --overwrite: $(args["overwrite"]). Use true/false.")
    end

    process_all_directories(args["simulation_root"], overwrite)
end