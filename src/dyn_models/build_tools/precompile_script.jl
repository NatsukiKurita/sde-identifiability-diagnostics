using HDF5
using StaticArrays
using DelaunayTriangulation
using LinearAlgebra
using Random

# --- 1. HDF5 の予行演習 ---
let
    path = tempname() * ".h5"
    h5open(path, "w") do file
        file["data"] = rand(10, 2) #
    end
    h5read(path, "data") # 読み込み
end

# --- 2. 計算の予行演習 ---
let
    v = [@SVector[randn(), randn()] for _ in 1:10]
    d = norm.(v) # LinearAlgebraの関数など
end

# --- 3. 近傍探索の予行演習 (追加推奨) ---
let
    points = rand(2, 5) # 5点のダミーデータ
    tri = triangulate(points)
end