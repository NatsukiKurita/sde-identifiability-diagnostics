module WA_TK2020
    using StaticArrays
    using LinearAlgebra
    using ..Geometry: project_onto_line_segment, AbstractWall, RectWall, CircleWall, NoWall

    import ..Types: AbstractWallAvoidance, AbstractState, AbstractBuffer,
            allocate_wall_buffers, reset_buffers!, calculate_wall_avoidance!

    export WA_TK2020

    struct WA_TK2020{T<:AbstractFloat} <: AbstractWallAvoidance
        k_wrep::T
        k_wcoh::T
    end

    # --- バッファの定義・関数 ---
    struct Buffer_WA_TK2020{T <: AbstractFloat} <: AbstractBuffer
        wall_nearest_points::Vector{SVector{2, T}}
        wall_smallest_distances::Vector{T}
        wall_nearest_normals::Vector{SVector{2, T}}
        wall_factor::Vector{T}
    end

    function allocate_wall_buffers(wall_avoidance::WA_TK2020, num_agents::Int, T::Type{<:AbstractFloat})
        wall_buffers = Buffer_WA_TK2020{T}(
            fill(SVector{2, T}(Inf, Inf), num_agents),
            fill(Inf, num_agents),
            fill(SVector{2, T}(zero(T), zero(T)), num_agents),
            fill(zero(T), num_agents)
        )
        return wall_buffers
    end

    function reset_buffers!(buffers::Buffer_WA_TK2020{T}) where T <: AbstractFloat
        fill!(buffers.wall_nearest_points, SVector{2, T}(Inf, Inf))
        fill!(buffers.wall_smallest_distances, Inf)
        fill!(buffers.wall_nearest_normals, SVector{2, T}(zero(T), zero(T)))
        fill!(buffers.wall_factor, zero(T))
    end

    # --- バッファの更新 ---
    function _update_wall_collision_info!(
        states,
        buffers,
        wall::RectWall{T},
        safe_eps::T
        ) where T <: AbstractFloat

        num_agents = length(states.positions)

        for i in 1:num_agents
            pos_i = states.positions[i]

            min_dist = T(Inf)
            nearest_point = SVector{2, T}(Inf, Inf)
            normal_at_nearest = SVector{2, T}(zero(T), zero(T))

            for (p_start, p_end) in wall.edges
                orth_proj, distance, normal_vec = project_onto_line_segment(pos_i, p_start, p_end, safe_eps)

                if distance < min_dist
                    min_dist = distance
                    nearest_point = orth_proj
                    normal_at_nearest = normal_vec
                end
            end

            buffers.wall_nearest_points[i] = nearest_point
            buffers.wall_smallest_distances[i] = min_dist
            buffers.wall_nearest_normals[i] = normal_at_nearest
        end
    end

    function _update_wall_collision_info!(
        states,
        buffers,
        wall::CircleWall{T},
        safe_eps::T
        ) where T <: AbstractFloat

        num_agents = length(states.positions)
        center = wall.center
        radius = wall.radius

        @inbounds for i in 1:num_agents
            pos_i = states.positions[i]

            vec_from_center = pos_i - center
            dist_from_center = norm(vec_from_center)
            dist_to_wall = radius - dist_from_center

            if dist_from_center > safe_eps
                normal_vec = -vec_from_center / dist_from_center
                nearest_point = center + (vec_from_center / dist_from_center) * radius
            else
                normal_vec = SVector{2, T}(zero(T), zero(T))
                nearest_point = center + SVector{2, T}(radius, zero(T))
            end

            buffers.wall_nearest_points[i] = nearest_point
            buffers.wall_smallest_distances[i] = dist_to_wall
            buffers.wall_nearest_normals[i] = normal_vec
        end
    end

    function _update_wall_collision_info!(
        states,
        buffers,
        wall::NoWall{T},
        safe_eps::T
        ) where T <: AbstractFloat

        num_agents = length(states.positions)
        @inbounds for i in 1:num_agents
            buffers.wall_nearest_points[i] = SVector{2, T}(Inf, Inf)
            buffers.wall_smallest_distances[i] = T(Inf)
            buffers.wall_nearest_normals[i] = SVector{2, T}(zero(T), zero(T))
        end
    end

    function calculate_wall_avoidance!(
        kinetics,
        wa::WA_TK2020{T},
        states,
        wall_buffers::Buffer_WA_TK2020{T},
        wall::AbstractWall,
        safe_eps::T
        ) where T <: AbstractFloat

        _update_wall_collision_info!(states, wall_buffers, wall, safe_eps)

        num_agents = length(states.positions)
        k_wrep = wa.k_wrep
        k_wcoh = wa.k_wcoh

        for i in 1:num_agents
            d_i = wall_buffers.wall_smallest_distances[i]
            n_i = wall_buffers.wall_nearest_normals[i]

            s, c = sincos(states.orientations[i])
            dir_sign = sign(dot(n_i, SVector{2, T}(c, s)))

            if dir_sign < 0
                magnitude = k_wrep
            elseif dir_sign > 0
                magnitude = k_wcoh
            else
                magnitude = zero(eltype(d_i))
            end

            wall_force = magnitude * n_i
            wall_buffers.wall_factor[i] = dot(wall_force, SVector{2, T}(-s, c))
        end
    end
end
