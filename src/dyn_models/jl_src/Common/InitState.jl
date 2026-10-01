module InitState
    using StableRNGs
    using StaticArrays
    using Random

    using ..Types: AbstractKinetics, allocate_states
    using ..Geometry: AbstractWall, RectWall, CircleWall, NoWall

    export generate_initial_positions, generate_initial_states

    # --- generate initial_positions for simulation ---
    # For CircleWall
    function generate_initial_positions(
        wall::CircleWall{T},
        num_agents::Int,
        rng::AbstractRNG
    ) where T <: AbstractFloat
        
        positions = Vector{SVector{2, T}}(undef, num_agents)
        for i in 1:num_agents
            # Uniform distribution inside the circle (with buffer)
            r = wall.radius * 0.8 * sqrt(rand(rng, T))
            θ = 2 * T(π) * rand(rng, T)
            positions[i] = wall.center + SVector(r*cos(θ), r*sin(θ))
        end
        return positions
    end

    # For RectWall
    function generate_initial_positions(
        wall::RectWall{T},
        num_agents::Int,
        rng::AbstractRNG
    ) where T <: AbstractFloat

        positions = Vector{SVector{2, T}}(undef, num_agents)

        # Generate in skewed coordinates
        origin = wall.corners[1]
        edge_u = wall.corners[2] - wall.corners[1]
        edge_v = wall.corners[4] - wall.corners[1]

        scale = T(0.8)
        offset = (T(1.0) - scale) / T(2.0)

        for i in 1:num_agents
            r1 = rand(rng, T)
            r2 = rand(rng, T)
            positions[i] = origin + (offset + scale * r1) * edge_u + (offset + scale * r2) * edge_v
        end
        return positions
    end

    # For NoWall
    function generate_initial_positions(
        wall::NoWall{T},
        num_agents::Int,
        rng::AbstractRNG
    ) where T <: AbstractFloat

        positions = Vector{SVector{2, T}}(undef, num_agents)
        for i in 1:num_agents
            positions[i] = SVector(rand(rng, T), rand(rng, T))
        end
        return positions
    end

    # --- 向き, 速度の初期化 ---
    # --- 初期化関数 ---
    function generate_initial_states(
        num_agents::Int,
        kinetics::AbstractKinetics,
        wall::AbstractWall,
        rng::AbstractRNG,
        T::Type{<:AbstractFloat}
    )

        # 1. 状態の割り当て
        states = allocate_states(kinetics, num_agents)

        # 初期位置の生成
        try
            initial_positions = generate_initial_positions(wall, num_agents, rng)
            states.positions .= initial_positions
        catch e
            error("Failed to generate initial positions: $(e)\n
                Wall type $(typeof(wall)) may not be supported for initialization.")
        end

        # 3. 向き、速度、角速度の初期化
        @inbounds for i in 1:num_agents
            # 向き [-π, π]
            states.orientations[i] = 2 * T(π) * rand(rng, T) - T(π)

            s, c = sincos(states.orientations[i])
            states.velocities[i] = SVector{2, T}(c, s) * kinetics.v_0
            
            # 角速度
            if hasproperty(kinetics, :ang_vel)
                states.ang_vel[i] = zero(T)
            end
        end

        return states
    end
end # module InitState