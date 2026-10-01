module Neighbor_Voronoi
    using DelaunayTriangulation
    using Random

    import ..Types: AbstractNeighborMethod, AbstractState, AbstractBuffer,
        allocate_neighbor_buffers, update_neighbors!

    export Neighbor_Voronoi, NeighborBuffer_Voronoi

    struct Neighbor_Voronoi <: AbstractNeighborMethod
        # Voronoi隣接はパラメータなし。隣接関係はドロネー三角形分割から得る。
    end

    struct NeighborBuffer_Voronoi{T<:AbstractFloat} <: AbstractBuffer
        weights::Matrix{T}
    end

    function allocate_neighbor_buffers(neighbor_method::Neighbor_Voronoi, num_agents::Int, T::Type{<:AbstractFloat})
        weights = zeros(T, num_agents, num_agents)
        return NeighborBuffer_Voronoi(weights)
    end

    function update_neighbors!(
        neighbor_method::Neighbor_Voronoi,
        state::AbstractState,
        neighbor_buffer::NeighborBuffer_Voronoi,
        rng::AbstractRNG
    )

        positions = state.positions
        N = length(positions)

        neighbor_buffer.weights .= 0

        if N <= 0
            println("No agents to compute neighbors for.")
            return
        elseif N == 1
            return
        elseif N == 2
            neighbor_buffer.weights[1, 2] = 1
            neighbor_buffer.weights[2, 1] = 1
            return
        end

        # ドロネー辺は Voronoi 隣接に一致する
        tri = triangulate(positions, rng=rng)
        for i in 1:length(positions)
            for j in get_neighbours(tri, i)
                if j > 0 && j <= length(positions) && j != i
                    neighbor_buffer.weights[i, j] = 1
                end
            end
            N_neighbors = sum(neighbor_buffer.weights[i, :])
            if N_neighbors != 0
                neighbor_buffer.weights[i, :] ./= N_neighbors
            end
        end
    end
end
