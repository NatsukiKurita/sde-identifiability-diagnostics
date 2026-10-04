module ConfigParser
    using TOML
    using ..Types

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

        sim_cfg    = config["Simulation"]
        precision_str = lowercase(string(get(sim_cfg, "precision", "Float64")))
        T = if precision_str == "float32"
            Float32
        elseif precision_str == "float64"
            Float64
        else
            error("Unsupported precision: $(precision_str). Use 'Float32' or 'Float64'.")
        end
        num_frames = Int(sim_cfg["num_frames"])
        dt         = T(sim_cfg["dt"])
        safe_eps   = T(sim_cfg["safe_eps"])

        config["Kinetics"]["v_0"] = T(sim_cfg["v_0"]) # v_0 is a kinetics parameter, so add it to config here so it can be set on the kinetics struct

        kinetics       = build_component(config["Kinetics"], T)
        social         = build_component(config["Social"], T)

        return kinetics, social, num_frames, dt, safe_eps, T
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
            elseif key == "v_0" # v_0 is a kinetics parameter and is set on the kinetics struct; skip it here and set it later.
                push!(args, T(0.0)) # Insert a placeholder value; the real value is set on the kinetics struct later
             else
                error("Missing required parameter '$name' for component type '$type_name'")
            end
        end
        return ComponentType(args...)
    end
end