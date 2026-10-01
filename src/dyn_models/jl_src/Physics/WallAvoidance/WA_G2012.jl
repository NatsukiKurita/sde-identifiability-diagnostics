module WA_G2012
    using StaticArrays
    using LinearAlgebra
    using ..Geometry: ray_casting_onto_line_segment, AbstractWall, RectWall, CircleWall, NoWall

    import ..Types: AbstractWallAvoidance, AbstractState, AbstractBuffer, 
            allocate_wall_buffers, reset_buffers!, calculate_wall_avoidance!

    export WA_G2012

    struct WA_G2012{T<:AbstractFloat} <: AbstractWallAvoidance
        k_w::T
    end

    # --- バッファの定義・関数 ---
    mutable struct Buffer_WA_G2012{T <: AbstractFloat} <: AbstractBuffer
        wall_collision_points::Vector{SVector{2, T}}
        wall_collision_distances::Vector{T}
        wall_collision_normals::Vector{SVector{2, T}}
        wall_factor::Vector{T}
        has_warned_no_collision::Bool
    end

    function allocate_wall_buffers(wall_avoidance::WA_G2012, num_agents::Int, T::Type{<:AbstractFloat}) 
        wall_buffers = Buffer_WA_G2012{T}(
            fill(SVector{2, T}(Inf, Inf), num_agents),
            fill(Inf, num_agents),
            fill(SVector{2, T}(zero(T), zero(T)), num_agents),
            zeros(T, num_agents),
            false
        )
        return wall_buffers
    end

    function reset_buffers!(buffers::Buffer_WA_G2012{T}) where T <: AbstractFloat
        num_agents = length(buffers.wall_collision_points)
        fill!(buffers.wall_collision_points, SVector{2, T}(Inf, Inf))
        fill!(buffers.wall_collision_distances, Inf)
        fill!(buffers.wall_collision_normals, SVector{2, T}(zero(T), zero(T)))
        fill!(buffers.wall_factor, zero(T))
    end

    # --- バッファの更新 ---
    # 矩形壁
    function _update_wall_collision_info!(
        state,
        buffers,
        wall::RectWall{T},
        safe_eps::T
        ) where T <: AbstractFloat

        num_agents = length(state.positions)

        for i in 1:num_agents
            pos_i = state.positions[i]
            theta_dir_i = state.orientations[i]
            
            # 初期化
            min_dist = T(Inf)
            nearest_point = SVector{2, T}(Inf, Inf)
            normal_at_nearest = SVector{2, T}(0.0, 0.0)

            for (p_start, p_end) in wall.edges
                hit_point, distance, normal_vec = ray_casting_onto_line_segment(pos_i, p_start, p_end, theta_dir_i, safe_eps)
                
                if distance < min_dist
                    min_dist = distance
                    nearest_point = hit_point
                    normal_at_nearest = normal_vec
                end
            end

            buffers.wall_collision_points[i] = nearest_point
            buffers.wall_collision_distances[i] = min_dist
            buffers.wall_collision_normals[i] = normal_at_nearest
        end
    end

    # 円形壁との衝突位置計算
    function _update_wall_collision_info!(
        state,
        buffers,
        wall::CircleWall,
        safe_eps::T
        ) where T <: AbstractFloat

        center = wall.center
        radius = wall.radius
        num_agents = length(state.positions)

        for i in 1:num_agents
            pos = state.positions[i]
            sin_val, cos_val = sincos(state.orientations[i])
            dir = SVector(cos_val, sin_val)

            # 交点を求めるために二次方程式を解く
            oc = pos - center
            a = T(1.0) # dot(dir, dir) == 1.0
            b = T(2.0) * dot(oc, dir)
            c = dot(oc, oc) - radius^2
            discriminant = b^2 - 4*a*c

            if discriminant < 0
                buffers.wall_collision_points[i] = SVector{2, eltype(pos)}(Inf, Inf)
                buffers.wall_collision_distances[i] = Inf
                continue
            end

            sqrt_disc = sqrt(discriminant)
            t1 = (-b - sqrt_disc) / (2*a)
            t2 = (-b + sqrt_disc) / (2*a)

            t_collision = Inf
            if t1 >=0 && t1 < t_collision
                t_collision = t1
            end
            if t2 >=0 && t2 < t_collision
                t_collision = t2
            end

            if t_collision == Inf
                buffers.wall_collision_points[i] = SVector{2, eltype(pos)}(Inf, Inf)
                buffers.wall_collision_distances[i] = Inf
                buffers.wall_factor[i] = zero(T)
                continue
            end

            collision_point = pos + t_collision * dir
            buffers.wall_collision_points[i] = collision_point
            buffers.wall_collision_distances[i] = t_collision

            normal = collision_point - center # 向きに注意, 外向きベクトル
            n_len = norm(normal)
            if n_len < safe_eps
                buffers.wall_collision_normals[i] = SVector{2, eltype(pos)}(0.0, 0.0)
            else
                buffers.wall_collision_normals[i] = normal / n_len
            end
        end
    end

    # NoWallの場合
    function _update_wall_collision_info!(
        state,
        buffers,
        wall::NoWall,
        safe_eps::T
        ) where T <: AbstractFloat

        num_agents = length(state.positions)
        for i in 1:num_agents
            buffers.wall_collision_points[i] = SVector{2, eltype(state.positions[i])}(Inf, Inf)
            buffers.wall_collision_distances[i] = Inf
            buffers.wall_collision_normals[i] = SVector{2, eltype(state.positions[i])}(0.0, 0.0)
        end
    end

    # --- 壁回避の最適角速度の更新 ---
    function calculate_wall_avoidance!(
        kinetics,
        wa::WA_G2012{T},
        states,
        wall_buffers,
        wall::AbstractWall,
        safe_eps::T,
    ) where T <: AbstractFloat

        _update_wall_collision_info!(states, wall_buffers, wall, safe_eps)

        num_agents = length(states.positions)
        k_w = wa.k_w

        for i in 1:num_agents
            # 壁との反発相互作用
            d_i_w = wall_buffers.wall_collision_distances[i]
            n_i_w = wall_buffers.wall_collision_normals[i]

            if d_i_w == Inf #壁が存在しない時のはず
                if !(wall isa NoWall) && !wall_buffers.has_warned_no_collision # wallが存在する場合にInfになるのはおかしい
                    wall_buffers.has_warned_no_collision = true
                    @warn "Collision distance is Inf for agent $i despite wall being present."
                end
            else
                delta_phi_i_w = states.orientations[i] - atan(n_i_w[2], n_i_w[1])
                phi_i_w = atan(sin(delta_phi_i_w), cos(delta_phi_i_w)) # -π to π
                wall_buffers.wall_factor[i] = k_w * sign(phi_i_w) / max(d_i_w, safe_eps)
            end
        end
    end
end