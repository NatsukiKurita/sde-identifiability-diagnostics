module InitState
    using StableRNGs
    using StaticArrays
    using Random

    using ..Types: AbstractKinetics, allocate_states

    export generate_initial_positions, generate_initial_states

    # 初期位置は開放領域上の初期雲。壁による拘束ではない。
    function generate_initial_positions(
        num_agents::Int,
        rng::AbstractRNG,
        T::Type{<:AbstractFloat}
    )
        positions = Vector{SVector{2, T}}(undef, num_agents)
        for i in 1:num_agents
            positions[i] = SVector{2, T}(rand(rng, T), rand(rng, T))
        end
        return positions
    end

    function generate_initial_states(
        num_agents::Int,
        kinetics::AbstractKinetics,
        rng::AbstractRNG,
        T::Type{<:AbstractFloat}
    )
        states = allocate_states(kinetics, num_agents)
        states.positions .= generate_initial_positions(num_agents, rng, T)

        @inbounds for i in 1:num_agents
            states.orientations[i] = 2 * T(π) * rand(rng, T) - T(π)

            s, c = sincos(states.orientations[i])
            states.velocities[i] = SVector{2, T}(c, s) * kinetics.v_0

            if hasproperty(states, :ang_vel)
                states.ang_vel[i] = zero(T)
            end
        end

        return states
    end
end # module InitState
