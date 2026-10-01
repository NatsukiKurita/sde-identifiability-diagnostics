module Kinetics_Overdamped
    using StaticArrays
    using LinearAlgebra
    using Random

    import ..Types: AbstractKinetics, AbstractState, AbstractBuffer,
        allocate_states, reset_buffers!,
        update_state_by_kinetics!, copy_state!

    export Kinetics_Overdamped, State_Overdamped

    # model definition
    struct Kinetics_Overdamped{T <: AbstractFloat} <: AbstractKinetics
        v_0::T
        sigma_theta::T
    end

    struct State_Overdamped{T <: AbstractFloat} <: AbstractState
        positions::Vector{SVector{2, T}}
        velocities::Vector{SVector{2, T}}
        orientations::Vector{T}
    end

    function allocate_states(kinetics::Kinetics_Overdamped{T}, num_agents::Int) where T <: AbstractFloat
        states = State_Overdamped{T}(
            fill(SVector{2, T}(zero(T), zero(T)), num_agents),
            fill(SVector{2, T}(zero(T), zero(T)), num_agents),
            zeros(T, num_agents)
        )
        return states
    end

    function update_state_by_kinetics!(
        kinetics::Kinetics_Overdamped{T},
        states::State_Overdamped{T},
        social_buffers::AbstractBuffer,
        dt::Real,
        rng::AbstractRNG
    ) where T <: AbstractFloat

        v_0 = kinetics.v_0
        drift_factor = dt                 # Precompute for efficiency
        noise_factor = kinetics.sigma_theta * sqrt(dt)   # Precompute for efficiency

        @inbounds @simd for i in eachindex(states.positions)
            # 更新前の情報
            current_orient = states.orientations[i]
            current_s, current_c = sincos(current_orient)
            current_direction = SVector{2, T}(current_c, current_s)

            # 力の計算
            total_factor = social_buffers.social_factor[i]

            # 角度の更新 (Euler-Maruyama法)
            states.orientations[i] += total_factor * drift_factor + noise_factor * randn(rng, T)

            # 速度と位置の更新 (Heun法)
            next_orient = states.orientations[i]
            next_s, next_c = sincos(next_orient)
            next_direction = SVector{2, T}(next_c, next_s)
            states.velocities[i] = v_0 * next_direction
            states.positions[i] += v_0 * (current_direction + next_direction) * 0.5 * dt
        end
    end

    function copy_state!(dest::State_Overdamped{T}, src::State_Overdamped{T}) where T <: AbstractFloat
        dest.positions    .= src.positions
        dest.velocities   .= src.velocities
        dest.orientations .= src.orientations
    end
end # module Kinetics_Overdamped