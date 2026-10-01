module SimRunner
    include("./Core/Types.jl")
    include("./Common/Geometry.jl")

    const PHYSICS_SUBDIRS = ("Kinetics", "Social", "WallAvoidance", "Neighbor")

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

    # 共通のモジュール
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
        if ispath(output_path) # 結果の存在確認
            if overwrite
                println("Warning: Output file already exists and will be overwritten: $output_path")
            else
                println("Note: Output file already exists and --overwrite is set to false. Skipping simulation: $output_path")
                return
            end
        end

        kinetics, social, wall_avoidance, neighbor, wall, num_frames, dt, safe_eps, T = load_simulation_setup(input_config)
        seed = sim_num  # sim_numをシードとして使用することで、同じsim_numであれば同じ初期状態と同じランダムな挙動が再現されるようにする

        println("Configuration successfully loaded. Starting simulation...\n")
        println("Number of agents: $agent_num, Number of frames: $num_frames, Time step: $dt, Seed: $seed")
        println("Kinetics            : ", nameof(typeof(kinetics)))
        println("Social Interactions : ", nameof(typeof(social)))
        println("Wall Avoidance      : ", nameof(typeof(wall_avoidance)))
        println("Neighbor Method     : ", nameof(typeof(neighbor)), "\n")

        rng = StableRNG(seed)

        @time history = run_simulation(
            kinetics, social, wall_avoidance, neighbor, 
            wall, agent_num, num_frames, dt, safe_eps, rng, T
        )

        save_simulation_h5(output_path, history)
        println("Results would be saved to: ", output_path)
    end
end