module Soc_G2012
    using StaticArrays
    using LinearAlgebra

    import ..Types: AbstractSocial, AbstractState, AbstractBuffer, 
            allocate_social_buffers, reset_buffers!, calculate_social_interactions!

    export Soc_G2012

    struct Soc_G2012{T<:AbstractFloat} <: AbstractSocial
        k_pos::T
        k_alg::T
    end

    # --- Buffer definitions and functions ---
    struct Buffer_Soc_G2012{T <: AbstractFloat} <: AbstractBuffer
        social_factor::Vector{T}
    end

    function allocate_social_buffers(social::Soc_G2012, num_agents::Int, T::Type{<:AbstractFloat})
        social_buffers = Buffer_Soc_G2012{T}(zeros(T, num_agents))
        return social_buffers
    end

    function reset_buffers!(buffers::Buffer_Soc_G2012{T}) where T <: AbstractFloat
        fill!(buffers.social_factor, zero(T))
    end

    # --- Update the social part of the optimal angular velocity ---
    function calculate_social_interactions!(
        kinetics,
        social::Soc_G2012,
        states,
        social_buffers,
        neighbor_buffers,
        safe_eps::T,
    ) where T <: AbstractFloat

        num_agents = length(states.positions)
        @inbounds @simd for i in 1:num_agents
            pos_i = states.positions[i]
            ori_i = states.orientations[i]
            social_interaction = zero(T)

            for j in 1:num_agents
                if i != j && neighbor_buffers.weights[i, j] > 0
                    pos_j = states.positions[j]

                    # Attraction
                    r_ij = pos_j - pos_i
                    d_ij = norm(r_ij)
                    phi_ij = atan(r_ij[2], r_ij[1]) - ori_i
                    attraction = social.k_pos * d_ij * sin(phi_ij)

                    # Alignment
                    ori_j = states.orientations[j]
                    delta_theta_ij = ori_j - ori_i
                    alignment = social.k_alg * sin(delta_theta_ij)

                    # Sum
                    social_interaction += (attraction + alignment) * neighbor_buffers.weights[i, j]
                end
            end
            social_buffers.social_factor[i] = social_interaction
        end
    end
end