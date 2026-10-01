module WA_Null
    using StaticArrays
    using ..Types

    import ..Types: AbstractWallAvoidance, AbstractState, AbstractBuffer,
            allocate_wall_buffers, reset_buffers!, calculate_wall_avoidance!

    export WA_Null

    struct WA_Null <: AbstractWallAvoidance
        # 壁との相互作用なし
    end

    struct Buffer_WA_Null{T <: AbstractFloat} <: AbstractBuffer
        wall_factor::Vector{T}
    end

    function allocate_wall_buffers(wall_avoidance::WA_Null, num_agents::Int, T::Type{<:AbstractFloat})
        return Buffer_WA_Null{T}(zeros(T, num_agents))
    end

    function reset_buffers!(buffers::Buffer_WA_Null{T}) where T <: AbstractFloat
        fill!(buffers.wall_factor, zero(T))
    end

    function calculate_wall_avoidance!(
        kinetics,
        wa::WA_Null,
        states,
        wall_buffers::Buffer_WA_Null{T},
        wall,
        safe_eps::T,
    ) where T <: AbstractFloat
        num_agents = length(states.positions)
        @inbounds @simd for i in 1:num_agents
            wall_buffers.wall_factor[i] = zero(T)
        end
    end
end
