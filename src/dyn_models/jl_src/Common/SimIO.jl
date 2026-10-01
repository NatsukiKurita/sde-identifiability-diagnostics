module SimIO
    using HDF5
    using StaticArrays
    using ArgParse

    import ..Geometry: AbstractWall

    export get_base_settings, get_float_type
    export create_boundary_info, save_simulation_h5

    # --- get parameter settings from argparse ---
    # ※ 既存のシェルスクリプトや実行環境を壊さないよう、引数群はそのまま残しています
    """
    function get_base_settings()
        s = ArgParseSettings()

        @add_arg_table! s begin
            "--output"
                help = "Output HDF5 file path"
                required = true
            "--num_agents"
                arg_type = Int
                required = true
            "--num_frames"
                arg_type = Int
                required = true
            "--dt"
                arg_type = Float64
                required = true
            "--boundary_shape"
                help = "Wall boundary type"
                required = true
            "--corners"
                arg_type = Float64
                nargs = '*'  # 可変長引数
                default = []
            "--center"
                arg_type = Float64
                nargs = '*'  # 可変長引数
                default = []
            "--radius"
                arg_type = Float64
                default = 0.0
            "--neighbor_method"
                help = "Neighbor search method"
                required = true
            "--cutoff_radius"
                arg_type = Float64
                default = nothing
            "--seed"
                arg_type = Int
                required = true
            "--precision"
                help = "Floating point precision (Float32 or Float64)"
                default = "Float64"
            "--safe_eps"
                arg_type = Float64
                default = 1e-8
        end
        return s
    end
    """

    # --- get floating point type from string ---
    function get_float_type(args::Dict)
        p = lowercase(get(args, "precision", "float64"))
        if p == "float32"
            return Float32
        elseif p == "float64"
            return Float64
        else
            error("Unsupported precision: $p. Use 'float32' or 'float64'.")
        end
    end

    # --- 境界情報出力 ---
    # will be deprecated 
    # 具体的な型（RectWall等）に依存せず、プロパティの有無で情報を抽出（ダックタイピング）
    function create_boundary_info(wall::AbstractWall)
        info = Dict{String, Any}(
            "boundary_shape" => string(nameof(typeof(wall)))
        )
        
        if hasproperty(wall, :corners)
            info["corners"] = wall.corners
        elseif hasproperty(wall, :center) && hasproperty(wall, :radius)
            info["center"] = wall.center
            info["radius"] = wall.radius
        end
        
        return info
    end

    # --- HDF5保存 ---
    function svector_to_matrix(vec_of_svec::AbstractArray{SVector{D, T}}) where {D, T}
        return reinterpret(reshape, T, vec_of_svec)
    end

    function save_simulation_h5(
        filename::String,
        state_hist, # NamedTuple等で :pos, :vel, :ori, :ang_vel を持つ履歴データ
    ) 

        h5open(filename, "w") do h5file
    
            # --- 状態変数・パラメータ情報 ---
            # Julia側の変数名(Symbol) -> HDF5(Python)側のデータセット名(String) の変換マップ
            state_mapping = [
                (:positions, "positions"),
                (:velocities, "velocities"),
                (:orientations, "orientations"),
                (:ang_vel, "angular_velocities")
            ]

            if !(hasproperty(state_hist, :positions))
                error("state_hist must contain at least a 'positions' field.")
            end

            for (jl_sym, h5_name) in state_mapping
                if hasproperty(state_hist, jl_sym)
                    val = getproperty(state_hist, jl_sym)
                    if val isa AbstractArray{<:SVector}
                        write(h5file, h5_name, svector_to_matrix(val))
                    else
                        write(h5file, h5_name, val)
                    end
                end
            end
        end
        println("Saved HDF5: $filename")
    end
end # module SimIO