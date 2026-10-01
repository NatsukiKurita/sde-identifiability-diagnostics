module Neighbor_Delaunay
    using DelaunayTriangulation
    using Random

    import ..Types: AbstractNeighborMethod, AbstractState, AbstractBuffer, 
        allocate_neighbor_buffers, update_neighbors!

    export Neighbor_Delaunay, NeighborBuffer_Delaunay

    struct Neighbor_Delaunay <: AbstractNeighborMethod
        # Delaunay法はパラメータなし
    end

    struct NeighborBuffer_Delaunay{T<:AbstractFloat} <: AbstractBuffer
        weights::Matrix{T} # 各エージェントの近傍エージェントのインデックスを格納する密行列
    end

    function allocate_neighbor_buffers(neighbor_method::Neighbor_Delaunay, num_agents::Int, T::Type{<:AbstractFloat})
        weights = zeros(T, num_agents, num_agents) # 各エージェントの近傍エージェントのインデックスを格納する密行列
        return NeighborBuffer_Delaunay(weights)
    end

    # 密行列に統一
    function update_neighbors!(
        neighbor_method::Neighbor_Delaunay,
        state::AbstractState,
        neighbor_buffer::NeighborBuffer_Delaunay,
        rng::AbstractRNG
    )

        positions = state.positions
        N = length(positions)

        neighbor_buffer.weights .= 0 # 近傍行列を初期化

        if N <= 0
            println("No agents to compute neighbors for.")
            return
        elseif N == 1
            return
        elseif N == 2
            # 2個体の場合は互いに近傍
            neighbor_buffer.weights[1, 2] = 1
            neighbor_buffer.weights[2, 1] = 1 
            return
        end

        # 3個体以上の場合にドロネー三角形分割を計算
        tri = triangulate(positions, rng=rng)
        for i in 1:length(positions)
            for j in get_neighbours(tri, i)
                if j > 0 && j <= length(positions) && j != i
                    neighbor_buffer.weights[i, j] = 1
                end
            end
            N_neighbors = sum(neighbor_buffer.weights[i, :])
            if N_neighbors != 0
                neighbor_buffer.weights[i, :] ./= N_neighbors # 正規化
            end
        end
    end
end

