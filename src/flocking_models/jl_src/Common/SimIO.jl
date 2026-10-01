module SimIO
    using HDF5
    using StaticArrays

    export get_float_type, save_simulation_h5

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