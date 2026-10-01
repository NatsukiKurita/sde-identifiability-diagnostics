module Dynamics_Inertial
    using StaticArrays
    using Random

    import ..Types: AbstractKinetics, AbstractState, AbstractBuffer,
        allocate_states, update_state_by_kinetics!, copy_state!

    export Dynamics_Inertial, State_Inertial

    # model definition
    struct Dynamics_Inertial{T <: AbstractFloat} <: AbstractKinetics
        v_0::T
        tau::T
        sigma_omega::T
    end

    struct State_Inertial{T <: AbstractFloat} <: AbstractState
        positions::Vector{SVector{2, T}}
        velocities::Vector{SVector{2, T}}
        orientations::Vector{T}
        ang_vel::Vector{T}
    end

    function allocate_states(kinetics::Dynamics_Inertial{T}, num_agents::Int) where T <: AbstractFloat
        states = State_Inertial{T}(
            fill(SVector{2, T}(zero(T), zero(T)), num_agents),
            fill(SVector{2, T}(zero(T), zero(T)), num_agents),
            zeros(T, num_agents),
            zeros(T, num_agents),
        )
        return states
    end

    function update_state_by_kinetics!(
        kinetics::Dynamics_Inertial{T},
        states::State_Inertial{T}, 
        social_buffers::AbstractBuffer,
        dt::Real,
        rng::AbstractRNG
    ) where T <: AbstractFloat
        
        dt_T = T(dt)
        drift_factor = dt_T / kinetics.tau
        noise_factor = kinetics.sigma_omega * sqrt(dt_T)

        @inbounds @simd for i in eachindex(states.ang_vel)
            current_ang_vel = states.ang_vel[i]
            current_ori = states.orientations[i]
            current_s, current_c = sincos(current_ori)
            current_direction = SVector{2, T}(current_c, current_s)
            target_ang_vel = social_buffers.social_factor[i]

            states.ang_vel[i] += -(current_ang_vel - target_ang_vel) * drift_factor + noise_factor * randn(rng, T)
            states.orientations[i] += T(0.5) * (current_ang_vel + states.ang_vel[i]) * dt_T

            next_s, next_c = sincos(states.orientations[i])
            next_direction = SVector{2, T}(next_c, next_s)

            states.velocities[i] = next_direction * kinetics.v_0
            states.positions[i] += T(0.5) * kinetics.v_0 * (current_direction + next_direction) * dt_T
        end
    end

    function copy_state!(dest::State_Inertial{T}, src::State_Inertial{T}) where T <: AbstractFloat
        dest.positions    .= src.positions
        dest.velocities   .= src.velocities
        dest.orientations .= src.orientations
        dest.ang_vel      .= src.ang_vel
    end
end # module Dynamics_Inertial