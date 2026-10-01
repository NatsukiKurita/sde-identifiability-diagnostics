module ConfigParser
    using TOML
    using ..Types
    using ..Geometry: construct_wall_from_boundary_info

    export load_simulation_setup, build_component, resolve_component_type

    function resolve_component_type(type_name::Symbol, parentmod::Module)
        if isdefined(parentmod, type_name)
            candidate = getproperty(parentmod, type_name)
            if candidate isa Module && isdefined(candidate, type_name)
                return getproperty(candidate, type_name)
            elseif candidate isa DataType
                return candidate
            end
        end

        for name in names(parentmod, all=true)
            submod = getproperty(parentmod, name)
            if submod isa Module && submod !== parentmod && isdefined(submod, type_name)
                return getproperty(submod, type_name)
            end
        end

        return nothing
    end

    function load_simulation_setup(file_path::String)
        config = TOML.parsefile(file_path)

        precision_str = get(config, "precision", "Float32")
        T = precision_str == "Float32" ? Float32 : Float64

        sim_cfg    = config["Simulation"]
        num_frames = Int(sim_cfg["num_frames"])
        dt         = T(sim_cfg["dt"])
        safe_eps   = T(sim_cfg["safe_eps"])

        config["Kinetics"]["v_0"] = T(sim_cfg["v_0"]) # v_0はkineticsのパラメータなので、kinetics構造体にセットするためにここでconfigに追加しておく

        kinetics       = build_component(config["Kinetics"], T)
        social         = build_component(config["Social"], T)
        wall_avoidance = build_component(config["WallAvoidance"], T)
        neighbor       = build_component(config["Neighbor"], T)

        boundary_cfg = config["Boundary"]
        wall = construct_wall_from_boundary_info(boundary_cfg, T)

        return kinetics, social, wall_avoidance, neighbor, wall, num_frames, dt, safe_eps, T
    end

    function build_component(config::Dict, T::DataType)
        type_name = Symbol(config["type"])
        parentmod = parentmodule(@__MODULE__)
        ComponentType = resolve_component_type(type_name, parentmod)

        if isnothing(ComponentType)
            error("Component type '$type_name' not found in any module.")
        end

        fnames = fieldnames(ComponentType)
        args = []
        for name in fnames
            key = string(name)
            if haskey(config, key)
                push!(args, T(config[key]))
            elseif key == "v_0" # v_0はkineticsのパラメータなので、kinetics構造体にセットする。ここではスキップして後でセットする。
                push!(args, T(0.0)) # ダミーの値を入れておいて、後でkinetics構造体にセットする
             else
                error("Missing required parameter '$name' for component type '$type_name'")
            end
        end
        return ComponentType(args...)
    end
end