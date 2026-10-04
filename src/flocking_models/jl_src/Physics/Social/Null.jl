module Soc_Null
    using StaticArrays
    using LinearAlgebra
    using ..Types

    import ..Types: AbstractSocial, AbstractState, AbstractBuffer, 
            allocate_social_buffers, reset_buffers!, calculate_social_interactions!

    export Soc_Null

    struct Soc_Null <: AbstractSocial
        # No social interaction in this model, so there are no parameters
    end

    struct Buffer_Soc_Null{T <: AbstractFloat} <: AbstractBuffer
        social_factor::Vector{T}
    end
    
    function allocate_social_buffers(social::Soc_Null, num_agents::Int, T::Type{<:AbstractFloat}) 
        social_buffers = Buffer_Soc_Null{T}(zeros(T, num_agents))
        return social_buffers
    end

    function reset_buffers!(buffers::Buffer_Soc_Null{T}) where T <: AbstractFloat
        fill!(buffers.social_factor, zero(T))
    end

    function calculate_social_interactions!(
        kinetics,
        social::Soc_Null,
        states,
        social_buffers,
        neighbor_buffers,
        safe_eps::T,
    ) where T <: AbstractFloat
        num_agents = length(states.positions)
        @inbounds @simd for i in 1:num_agents
            social_buffers.social_factor[i] = zero(T)
        end
    end
end # module Soc_Null