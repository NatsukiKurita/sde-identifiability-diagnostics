using StaticArrays
using DelaunayTriangulation
using ArgParse
using HDF5

function read_args()
    parser = ArgParseSettings()
    @add_arg_table parser begin
        "--data_h5_root"
            help = "Path to the root directory containing HDF5 files"
            arg_type = String
            required = true
    end
    return parse_args(parser)
end

function restore_neighbors(positions)
    dim, num_agents, num_steps = size(positions)
    
    neighbor_list = [ [Int[] for _ in 1:num_agents] for _ in 1:num_steps ]
    max_neighbors = 0

    if num_agents == 1
        # Do nothing
    elseif num_agents == 2
        for t in 1:num_steps
            push!(neighbor_list[t][1], 2)
            push!(neighbor_list[t][2], 1)
        end
        max_neighbors = 1
    else
        for t in 1:num_steps
            pts = [(positions[1, i, t], positions[2, i, t]) for i in 1:num_agents]
            tri = triangulate(pts)
            
            for i in 1:num_agents
                for j in get_neighbours(tri, i)
                    if j > 0 && j <= num_agents && j != i
                        push!(neighbor_list[t][i], j)
                    end
                end
                # Update the maximum number of neighbors
                if length(neighbor_list[t][i]) > max_neighbors
                    max_neighbors = length(neighbor_list[t][i])
                end
            end
        end
    end
    
    # Dimensions are (time steps, agents, max neighbors): T × N × E
    padded_neighbors = fill(0, (num_steps, num_agents, max_neighbors))
    
    for t in 1:num_steps
        for i in 1:num_agents
            neighbors = neighbor_list[t][i]
            for (k, neighbor_idx) in enumerate(neighbors)
                # Write in (t, i, k) order
                padded_neighbors[t, i, k] = neighbor_idx 
            end
        end
    end
    
    # Convert 0 -> -1 (padding) and 1-based -> 0-based indices
    padded_neighbors .= padded_neighbors .- 1 
    return permutedims(padded_neighbors, (3, 2, 1)) # Permute dimensions to match (T, N, E) on the Python side
end

if abspath(PROGRAM_FILE) == @__FILE__
    args = read_args()
    data_h5_root = args["data_h5_root"]

    target_files = String[]
    for (root, dirs, files) in walkdir(data_h5_root)
        for file in files
            if endswith(file, ".h5")
                push!(target_files, joinpath(root, file))
            end
        end
    end
    println("Found ", length(target_files), " HDF5 files in ", data_h5_root)

    dataset_name = "delaunay_neighbors"
    for file in target_files
        println("Processing: ", file)
        h5open(file, "r+") do h5

            # Skip if it already exists
            if haskey(h5, dataset_name)
                println("Dataset '", dataset_name, "' already exists in ", file, ". Skipping.")
            else
                positions = read(h5, "positions")
                padded_neighbors = restore_neighbors(positions)
                
                if isempty(padded_neighbors) || size(padded_neighbors, 1) == 0 || size(padded_neighbors, 2) == 0
                    # If empty, write without specifying chunk or deflate
                    h5[dataset_name] = padded_neighbors
                    println("Saved EMPTY Delaunay neighbors to: ", dataset_name)
                else
                    # Normal write
                    h5[dataset_name, deflate=4, chunk=(size(padded_neighbors, 1), size(padded_neighbors, 2), 1)] = padded_neighbors
                    println("Saved Delaunay neighbors to: ", dataset_name)
                end
            end
        end
    end
end
