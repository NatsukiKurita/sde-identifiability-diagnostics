module Geometry
    using StaticArrays
    using LinearAlgebra

    export AbstractWall, RectWall, CircleWall, NoWall
    export construct_wall_from_boundary_info
    export project_onto_line_segment, ray_casting_onto_line_segment

    # --- Wall structures ---
    # PolygonWallが実装されたら廃止するかも
    abstract type AbstractWall{T <: AbstractFloat} end

    struct RectWall{T <: AbstractFloat} <: AbstractWall{T}
        corners::SVector{4, SVector{2, T}}
        edges::SVector{4, Tuple{SVector{2, T}, SVector{2, T}}}
        x_min::T
        x_max::T
        y_min::T
        y_max::T

        function RectWall{T}(corners::SVector{4, SVector{2, T}}) where T <: AbstractFloat
            center = sum(corners) / T(4.0)
            sorted_indices = sortperm(corners, by = c -> atan(c[2] - center[2], c[1] - center[1]))
            c = corners[sorted_indices]
            
            edges = SVector{4, Tuple{SVector{2, T}, SVector{2, T}}}(
                (c[1], c[2]),
                (c[2], c[3]),
                (c[3], c[4]),
                (c[4], c[1]),
            )

            x_vals = map(p -> p[1], c)
            y_vals = map(p -> p[2], c)

            new{T}(c, edges, minimum(x_vals), maximum(x_vals), minimum(y_vals), maximum(y_vals))
        end
    end

    struct CircleWall{T <: AbstractFloat} <: AbstractWall{T}
        center::SVector{2, T}
        radius::T
    end

    struct NoWall{T <: AbstractFloat} <: AbstractWall{T} end


    # --- Wall construction from args dictionary ---
    function construct_wall_from_boundary_info(args::Dict, T::Type{<:AbstractFloat})
        boundary_shape = get(args, "boundary_shape", "Not Specified")
        if boundary_shape == "RectWall"
            raw_corners = args["corners"]
            if isempty(raw_corners)
                error("argument --corners is required for RectWall")
            elseif length(raw_corners) != 8
                error("argument --corners must contain exactly 8 elements for RectWall")
            end
            corners = SVector{4, SVector{2, T}}(
                SVector{2, T}(raw_corners[1], raw_corners[2]),
                SVector{2, T}(raw_corners[3], raw_corners[4]),
                SVector{2, T}(raw_corners[5], raw_corners[6]),
                SVector{2, T}(raw_corners[7], raw_corners[8]),
            )
            return RectWall{T}(corners)
        elseif boundary_shape == "CircleWall"
            center_coords = args["center"]
            radius = args["radius"]
            if isempty(center_coords)
                error("argument --center is required for CircleWall")
            elseif length(center_coords) != 2
                error("argument --center must contain exactly 2 elements for CircleWall")
            elseif radius === nothing
                error("argument --radius is required for CircleWall")
            end
            center = SVector{2, T}(center_coords[1], center_coords[2])
            return CircleWall{T}(center, radius)
        elseif boundary_shape == "NoWall"
            return NoWall{T}()
        else
            error("Unsupported boundary_shape: $boundary_shape")
        end
    end

    # --- Geometry utility functions ---
    # 点を線分に直交射影する関数
    function project_onto_line_segment(
        point::SVector{2, T}, 
        seg_start::SVector{2, T}, 
        seg_end::SVector{2, T},
        safe_eps::T
        ) where T <: AbstractFloat
        
        seg_vector = seg_end - seg_start
        point_vector = point - seg_start
        seg_length_squared = dot(seg_vector, seg_vector)
        if seg_length_squared < safe_eps
            distance = norm(point_vector)
            normal_vector = if distance > safe_eps
                point_vector / distance
            else
                SVector{2, T}(0.0, 0.0)
            end
            return (seg_start, distance, normal_vector)
        end

        t = dot(point_vector, seg_vector) / seg_length_squared
        t_clamped = clamp(t, zero(T), one(T))

        projected_point = seg_start + t_clamped * seg_vector
        distance = norm(point - projected_point)
        # 内向きの法線
        normal_vector = if distance > safe_eps
            (point - projected_point) / distance
        else
            SVector{2, T}(0.0, 0.0)
        end
        return (projected_point, distance, normal_vector)
    end

    function ray_casting_onto_line_segment(
        ray_origin::SVector{2, T},
        seg_start::SVector{2, T},
        seg_end::SVector{2, T},
        orient_theta::T,
        safe_eps::T
        ) where T <: AbstractFloat

        seg_vector = seg_end - seg_start
        point_vector = ray_origin - seg_start
        sin_val, cos_val = sincos(orient_theta)
        orientation = SVector{2, T}(cos_val, sin_val)

        denom = seg_vector[1] * orientation[2] - seg_vector[2] * orientation[1]
        if abs(denom) < safe_eps
            crossing_point = SVector{2, T}(Inf, Inf)
            distance = Inf
            normal_vector = SVector{2, T}(0.0, 0.0)
            return (crossing_point, distance, normal_vector)
        end

        s = (seg_vector[2] * point_vector[1] - seg_vector[1] * point_vector[2]) / denom
        t = (orientation[2] * point_vector[1] - orientation[1] * point_vector[2]) / denom

        if t >= zero(T) - safe_eps && s >= zero(T) - safe_eps && t <= one(T) + safe_eps
            crossing_point = ray_origin + s * orientation
            distance = s # orientationベクトルは単位ベクトルなのでそのまま距離
            # 外向きの法線を返す
            normal_vector = SVector(-seg_vector[2], seg_vector[1]) / max(norm(seg_vector), safe_eps)
            if dot(normal_vector, orientation) < zero(T)
                normal_vector = -normal_vector
            end
            return (crossing_point, distance, normal_vector)
        else
            crossing_point = SVector{2, T}(Inf, Inf)
            distance = Inf
            normal_vector = SVector{2, T}(0.0, 0.0)
            return (crossing_point, distance, normal_vector)
        end
    end 
end # module Geometry