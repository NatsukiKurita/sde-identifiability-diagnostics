module Soc_KV
    using StaticArrays
    using LinearAlgebra

    import ..Types: AbstractSocial, AbstractState, AbstractBuffer, 
            allocate_social_buffers, reset_buffers!, calculate_social_interactions!

    export Soc_KV

    struct Soc_KV{T<:AbstractFloat} <: AbstractSocial
        k_alg::T
    end

    # --- Buffer definitions and functions ---
    struct Buffer_Soc_KV{T <: AbstractFloat} <: AbstractBuffer
        social_factor::Vector{T}
    end

    function allocate_social_buffers(social::Soc_KV, num_agents::Int, T::Type{<:AbstractFloat}) 
        social_buffers = Buffer_Soc_KV{T}(zeros(T, num_agents))
        return social_buffers
    end

    function reset_buffers!(buffers::Buffer_Soc_KV{T}) where T <: AbstractFloat
        fill!(buffers.social_factor, zero(T))
    end

    # --- Update the social part of the optimal angular velocity ---
    function calculate_social_interactions!(
        kinetics,
        social::Soc_KV,
        states,
        social_buffers,
        neighbor_buffers,
        safe_eps::T,
    ) where T <: AbstractFloat

        num_agents = length(states.positions)
        @inbounds @simd for i in 1:num_agents
            ori_i = states.orientations[i]
            social_interaction = zero(T)

            for j in 1:num_agents
                if i != j && neighbor_buffers.weights[i, j] > 0
                    ori_j = states.orientations[j]
                    delta_theta_ij = ori_j - ori_i
                    social_interaction += sin(delta_theta_ij) * neighbor_buffers.weights[i, j]
                end
            end
            social_buffers.social_factor[i] = social.k_alg * social_interaction
        end
    end
end