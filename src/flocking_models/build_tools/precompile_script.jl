using HDF5
using StaticArrays
using DelaunayTriangulation
using LinearAlgebra
using StableRNGs

# HDF5 書き出し。SimIO.save_simulation_h5 は SVector 配列を reinterpret して write する。
let
    path = tempname() * ".h5"
    positions = [SVector{2, Float64}(randn(), randn()) for _ in 1:4, _ in 1:2]
    data = reinterpret(reshape, Float64, positions)
    h5open(path, "w") do file
        write(file, "positions", data)
    end
end

# Social 側は SVector{2} に norm を掛ける。
let
    norm(SVector(randn(), randn()))
end

# Neighbor_Voronoi は Vector{SVector{2}} を rng 付きで triangulate し、get_neighbours を読む。
let
    rng = StableRNG(1)
    positions = [SVector{2, Float64}(rand(rng), rand(rng)) for _ in 1:5]
    tri = triangulate(positions, rng=rng)
    for i in eachindex(positions)
        for _ in get_neighbours(tri, i)
        end
    end
end
