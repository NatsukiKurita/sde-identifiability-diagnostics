module SimRunner
    include("./Core/Types.jl")

    const PHYSICS_SUBDIRS = ("Kinetics", "Social", "Neighbor")

    function include_physics_modules!()
        physics_root = joinpath(@__DIR__, "Physics")
        for subdir in PHYSICS_SUBDIRS
            dir = joinpath(physics_root, subdir)
            isdir(dir) || continue
            for file in sort(filter(f -> endswith(f, ".jl"), readdir(dir)))
                include(joinpath("Physics", subdir, file))
            end
        end
    end

    include_physics_modules!()

    # Common modules
    include("./Common/InitState.jl")
    include("./Common/SimIO.jl")
    include("./Core/Simulation.jl")
    include("./Core/ConfigParser.jl")

    using StableRNGs
    using .ConfigParser: load_simulation_setup
    using .Simulation: run_simulation
    using .SimIO: save_simulation_h5

    function execute_simulation(
        input_config::String, 
        output_path::String, 
        agent_num::Int,
        sim_num::Int,
        overwrite::Bool
        )

        println("Loading simulation setup from: $input_config")
        if ispath(output_path) # Check whether the result already exists
            if overwrite
                println("Warning: Output file already exists and will be overwritten: $output_path")
            else
                println("Note: Output file already exists and --overwrite is set to false. Skipping simulation: $output_path")
                return
            end
        end

        kinetics, social, num_frames, dt, safe_eps, T = load_simulation_setup(input_config)
        neighbor = Neighbor_Voronoi.Neighbor_Voronoi()
        seed = sim_num  # Use sim_num as the seed so the same sim_num reproduces the same initial state and random behavior

        println("Configuration successfully loaded. Starting simulation...\n")
        println("Number of agents: $agent_num, Number of frames: $num_frames, Time step: $dt, Seed: $seed")
        println("Kinetics            : ", nameof(typeof(kinetics)))
        println("Social Interactions : ", nameof(typeof(social)))
        println("Neighbor Method     : ", nameof(typeof(neighbor)), "\n")

        rng = StableRNG(seed)

        @time history = run_simulation(
            kinetics, social, neighbor, agent_num, num_frames, dt, safe_eps, rng, T
        )

        save_simulation_h5(output_path, history)
        println("Results would be saved to: ", output_path)
    end
end