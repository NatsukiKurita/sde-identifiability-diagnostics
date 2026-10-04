using Pkg

project_dir = dirname(@__DIR__)
Pkg.activate(mktempdir()) # Use a temporary environment
Pkg.add("PackageCompiler")

using PackageCompiler

Pkg.activate(project_dir)
Pkg.instantiate()

create_sysimage(
    [:DelaunayTriangulation,
    :HDF5,
    :StaticArrays,
    :ArgParse,
    :StableRNGs
    ],
    sysimage_path = joinpath(project_dir, "build_tools/sys_simulation.so"),
    precompile_execution_file = joinpath(project_dir, "build_tools/precompile_script.jl")
)