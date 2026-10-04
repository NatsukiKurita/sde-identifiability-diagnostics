module Soc_KR2021
    using StaticArrays
    using LinearAlgebra
    
    import  ..Types: AbstractSocial, AbstractState, AbstractBuffer, 
            allocate_social_buffers, reset_buffers!, calculate_social_interactions!

    export Soc_KR2021

    struct Soc_KR2021{T<:AbstractFloat} <: AbstractSocial
        k_alg::T
        k_pos::T
        m::T
        d_eq::T
    end

    # --- Buffer definitions and functions ---
    struct Buffer_Soc_KR2021{T <: AbstractFloat} <: AbstractBuffer
        social_factor::Vector{T} # projected social force for Soc_KR2021
    end

    function allocate_social_buffers(social::Soc_KR2021, num_agents::Int, T::Type{<:AbstractFloat}) 
        social_buffers = Buffer_Soc_KR2021{T}(zeros(T, num_agents))
        return social_buffers
    end

    function reset_buffers!(buffers::Buffer_Soc_KR2021{T}) where T <: AbstractFloat
        fill!(buffers.social_factor, zero(T))
    end

    # --- Update social interactions ---
    function calculate_social_interactions!(
        kinetics,
        social::Soc_KR2021,
        states,
        social_buffers,
        neighbor_buffers,
        safe_eps::T,
    ) where T <: AbstractFloat

        num_agents = length(states.positions)
        k_alg = social.k_alg
        k_pos = social.k_pos
        m = social.m
        d_eq = social.d_eq

        for i in 1:num_agents
            # Variables used for the alignment force
            theta_i = states.orientations[i]
            s, c = sincos(theta_i)
            vel_i =  states.velocities[i]  

            # Variables used for the repulsion force
            pos_i = states.positions[i]

            force_by_neighbors = SVector{2, T}(0.0, 0.0)
            for j in 1:num_agents
                if i != j && neighbor_buffers.weights[i, j] > 0
                    pos_j = states.positions[j]
                    r_ij = pos_j - pos_i
                    d_ij = norm(r_ij)
                    e_ij = r_ij / (d_ij + safe_eps)

                    # Alignment force
                    vel_j = states.velocities[j]
                    delta_v = vel_j - vel_i
                    force_alg = k_alg * delta_v * neighbor_buffers.weights[i, j]

                    magnitude_pos = k_pos * tanh(m * (d_ij - d_eq))
                    force_pos = magnitude_pos * e_ij * neighbor_buffers.weights[i, j]

                    force_by_neighbors += (force_alg + force_pos)
                end
            end
            projected_force_total = dot(force_by_neighbors, SVector{2, T}(-s, c))
            social_buffers.social_factor[i] = projected_force_total / kinetics.v_0
        end
    end
end