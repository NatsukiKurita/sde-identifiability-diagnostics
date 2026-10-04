module Simulation
    using Random
    using StaticArrays
    using ..Types
    using ..SimIO
    using ..InitState: generate_initial_states

    export run_simulation

    function run_simulation(
        kinetics::AbstractKinetics,
        social::AbstractSocial,
        neighbor_method::AbstractNeighborMethod,
        num_agents::Int,
        num_frames::Int,
        dt::Real,
        safe_eps::Real,
        rng::AbstractRNG,
        T::Type{<:AbstractFloat}
    )

        # Initialize states and buffers
        states = generate_initial_states(num_agents, kinetics, rng,  T) # Initialize states (positions, velocities, orientations, etc.)

        neighbor_buffers = allocate_neighbor_buffers(neighbor_method, num_agents, T) 
        social_buffers = allocate_social_buffers(social, num_agents, T)

        history_pos = Matrix{SVector{2, T}}(undef, num_agents, num_frames)
        history_ori = Matrix{T}(undef, num_agents, num_frames)
        history_vel = Matrix{SVector{2, T}}(undef, num_agents, num_frames)

        has_ang_vel = hasproperty(states, :ang_vel)
        if has_ang_vel
            history_ang_vel = Matrix{T}(undef, num_agents, num_frames)
        end

        for step in 1:num_frames
            update_neighbors!(neighbor_method, states, neighbor_buffers, rng) # Update neighbor buffers
            reset_buffers!(social_buffers) # Reset social buffers
            calculate_social_interactions!(kinetics, social, states, social_buffers, neighbor_buffers, safe_eps) # v_0 is stored in kinetics
            update_state_by_kinetics!(kinetics, states, social_buffers, dt, rng)
            
            # Save states
            history_pos[:, step] .= states.positions
            history_ori[:, step] .= states.orientations
            history_vel[:, step] .= states.velocities
            if has_ang_vel
                history_ang_vel[:, step] .= states.ang_vel
            end
        end

        if has_ang_vel
            nested_history = (positions=history_pos, orientations=history_ori, velocities=history_vel, ang_vel=history_ang_vel)
        else
            nested_history = (positions=history_pos, orientations=history_ori, velocities=history_vel)
        end
        return nested_history
    end
end # module Simulation