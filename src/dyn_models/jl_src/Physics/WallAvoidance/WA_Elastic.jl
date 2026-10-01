module WA_Elastic
    using StaticArrays
    using LinearAlgebra
    using ..Geometry: project_onto_line_segment, AbstractWall, RectWall, CircleWall, NoWall

    import  ..Types: AbstractWallAvoidance, AbstractState, AbstractBuffer, 
            allocate_wall_buffers, reset_buffers!, calculate_wall_avoidance!

    export WA_Elastic

    struct WA_Elastic{T<:AbstractFloat} <: AbstractWallAvoidance
        k_w::T
        d_w::T
    end

    # --- バッファの定義・関数 ---
    struct Buffer_WA_Elastic{T <: AbstractFloat} <: AbstractBuffer
        wall_nearest_points::Vector{SVector{2, T}}
        wall_smallest_distances::Vector{T}
        wall_nearest_normals::Vector{SVector{2, T}}
        wall_factor::Vector{T}
    end

    function allocate_wall_buffers(wall_avoidance::WA_Elastic, num_agents::Int, T::Type{<:AbstractFloat}) 
        wall_buffers = Buffer_WA_Elastic{T}(
            fill(SVector{2, T}(Inf, Inf), num_agents),
            fill(Inf, num_agents),
            fill(SVector{2, T}(zero(T), zero(T)), num_agents),
            fill(zero(T), num_agents)
        )
        return wall_buffers
    end

    function reset_buffers!(buffers::Buffer_WA_Elastic{T}) where T <: AbstractFloat
        num_agents = length(buffers.wall_nearest_points)
        fill!(buffers.wall_nearest_points, SVector{2, T}(Inf, Inf))
        fill!(buffers.wall_smallest_distances, Inf)
        fill!(buffers.wall_nearest_normals, SVector{2, T}(zero(T), zero(T)))
        fill!(buffers.wall_factor, zero(T))
    end

    # --- バッファの更新 ---
    # 矩形壁
    function _update_wall_collision_info!(
        states,
        buffers,
        wall::RectWall{T},
        safe_eps::T
        ) where T <: AbstractFloat

        num_agents = length(states.positions)

        for i in 1:num_agents
            pos_i = states.positions[i]
            
            # 初期化
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

    # 円形壁
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
            
            # 中心から自分へのベクトル
            vec_from_center = pos_i - center
            dist_from_center = norm(vec_from_center)

            # --- 水槽(Boundary)としての計算 ---
            # 壁までの距離 = 半径 - 中心からの距離
            dist_to_wall = radius - dist_from_center
            
            if dist_from_center > safe_eps
                # 法線: 壁から自分へ向かう方向 = 中心へ向かう方向
                normal_vec = -vec_from_center / dist_from_center
                
                # 最近点: 中心から自分の方向に radius 進んだ点
                nearest_point = center + (vec_from_center / dist_from_center) * radius
            else
                # 中心と重なっている場合の特異点処理
                normal_vec = SVector{2, T}(zero(T), zero(T))
                nearest_point = center + SVector{2, T}(radius, zero(T)) # 適当な円周上の点
            end

            buffers.wall_nearest_points[i] = nearest_point
            buffers.wall_smallest_distances[i] = dist_to_wall
            buffers.wall_nearest_normals[i] = normal_vec
        end
    end

    # NoWallの場合は何もしない
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
        
    # --- 壁との反発計算 ---
    function calculate_wall_avoidance!(
        kinetics,
        wa::WA_Elastic{T},
        states,
        wall_buffers,
        wall::AbstractWall,
        safe_eps::T
        ) where T <: AbstractFloat

        # First, update collision info
        _update_wall_collision_info!(states, wall_buffers, wall, safe_eps)
        
        num_agents = length(states.positions)

        k_w = wa.k_w
        d_w = wa.d_w
        d_w_safe = max(d_w, safe_eps)
        
        for i in 1:num_agents
            d_i_w = wall_buffers.wall_smallest_distances[i]
            n_i = wall_buffers.wall_nearest_normals[i]

            # 逆距離型バリア関数。d_w より内側に入ったときだけ反発する。
            d_i_w_safe = max(d_i_w, safe_eps)
            magnitude = d_i_w < d_w ? k_w * max(inv(d_i_w_safe) - inv(d_w_safe), zero(T)) : zero(T)
            s, c = sincos(states.orientations[i])
            wall_buffers.wall_factor[i] = dot(magnitude * n_i, SVector{2, T}(-s, c)) / kinetics.v_0
        end 
    end
end