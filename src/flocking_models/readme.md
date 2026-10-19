# Adding a model

The simulator is modular: kinetics and social interactions can be added independently. Simulations run in an open field with no walls. `SimRunner.jl` automatically `include`s every `.jl` file under the `Physics/` subdirectories, so placing a file in the right directory is all that is needed on the Julia side. Neighbors are fixed to Voronoi adjacency and are not selected in the TOML. The neighbor buffer is still kept because social modules read `neighbor_buffers.weights`.

The Python side (presets in `src/experiment_configs/parameter_presets/`, estimators in `src/QMLE/estimator_modules/`, and optimizer bounds in `src/experiment_configs/estimation.py`) must be updated separately.

---

## Common conventions

### File locations

| Category | Directory |
|----------|-----------|
| Kinetics | `jl_src/Physics/Kinetics/` |
| Social   | `jl_src/Physics/Social/` |
| Neighbor | `jl_src/Physics/Neighbor/` |

`SimRunner.jl` includes the `Kinetics`, `Social`, and `Neighbor` directories in that order, and the `.jl` files within each directory in lexicographic order of file name.
Important: avoid `import` dependencies between Physics modules and keep each file self-contained.

### Module names, struct names, and the TOML `type`

`ConfigParser.build_component` resolves the type by assuming that **the module name and the parameter struct name are identical**. The TOML `type` field holds that name as a string.

| Category | Example file | Module / struct name | TOML `type` |
|----------|--------------|----------------------|-------------|
| Kinetics | `Overdamped.jl` | `Dynamics_Overdamped` | `"Dynamics_Overdamped"` |
| Social   | `KR2021.jl` | `Soc_KR2021` | `"Soc_KR2021"` |
| Neighbor | `Voronoi.jl` | `Neighbor_Voronoi` | Fixed (not written in the TOML) |

Naming rules:

- **Kinetics**: module and struct names use the `Dynamics_` prefix
- **Social**: module and struct names use the `Soc_` prefix
- **Neighbor**: module and struct names use the `Neighbor_` prefix. Only `Neighbor_Voronoi` is used at run time

### Parameter structs and the TOML

The **field names** of the parameter struct are the TOML keys (`ConfigParser` reads them with `fieldnames`). Every field must be present in the TOML. For example, `scripts/01_simulation/00_prepare_toml.py` writes the following section for KR2021 in `setting_1`:

```toml
[Social]
type = "Soc_KR2021"
k_alg = 1.0
k_pos = 1.0
m = 1.0
d_eq = 0.02
```

Exception: the kinetics `v_0` is injected from `v_0` in the `[Simulation]` section, so it is not written in the `[Kinetics]` section (the kinetics struct still needs a `v_0` field).

### Roles in the simulation loop

`Core/Simulation.jl` calls the following functions in this order at every step:

1. `update_neighbors!` — update the neighbor buffer
2. `reset_buffers!` — reset the social buffer
3. `calculate_social_interactions!` — write `social_factor`
4. `update_state_by_kinetics!` — update the state (positions, velocities, orientations, and so on)

The social module writes a **scalar factor** `social_factor` that the kinetics module reads, and the kinetics module uses it to update the state.

---

## Kinetics

**Reference implementations**: `Physics/Kinetics/Overdamped.jl`, `Inertial.jl`

### What to define

1. **Parameter struct** `Dynamics_<Name>{T} <: AbstractKinetics`
   - Required field: `v_0::T` (supplied from `[Simulation]` in the TOML)
   - Other parameters are free (for example `sigma_theta` for Overdamped, and `sigma_omega` and `tau` for Inertial)

2. **State struct** `State_<Name>{T} <: AbstractState` (for example `State_Overdamped`, `State_Inertial`)
   - Required fields: `positions::Vector{SVector{2, T}}`, `velocities::Vector{SVector{2, T}}`, `orientations::Vector{T}` (one entry per agent)
   - Angular-velocity models add `ang_vel::Vector{T}`. `Simulation.jl` checks `hasproperty(states, :ang_vel)` to record it, and `SimIO.jl` writes it to the HDF5 dataset `angular_velocities`

3. **Required functions**

   | Function | Role |
   |----------|------|
   | `allocate_states(kinetics, num_agents)` | Return a zero-initialized state |
   | `update_state_by_kinetics!(kinetics, states, social_buffers, dt, rng)` | Advance the state by one step |

   The existing modules also define `copy_state!(dest, src)`, but the simulator does not call it.

### Notes

- Initial conditions are set by `Common/InitState.jl`, not by the kinetics module: positions are uniform in \([0, 1)^2\), orientations are uniform in \([-\pi, \pi)\), velocities are `v_0` times the heading, and `ang_vel` (if present) is zero
- Read `social_buffers.social_factor[i]` and apply it to the orientation, angular velocity, or other state as the model requires
- The buffer is received as an `AbstractBuffer`, and the code relies on the field name `.social_factor` (an implicit interface contract)
- `import ..Types: AbstractKinetics, AbstractState, AbstractBuffer, allocate_states, update_state_by_kinetics!`

---

## Social interaction

**Reference implementations**: `Physics/Social/KR2021.jl`, `Null.jl`

### What to define

1. **Parameter struct** `Soc_<Name> <: AbstractSocial` (may have no parameters)

2. **Buffer struct** `Buffer_Soc_<Name>{T} <: AbstractBuffer`
   - **Required field**: `social_factor::Vector{T}` (length = number of agents)
   - The kinetics module reads it at every step

3. **Required functions**

   | Function | Role |
   |----------|------|
   | `allocate_social_buffers(social, num_agents, T)` | Create the buffer |
   | `reset_buffers!(buffers)` | Zero `social_factor` at the start of every step |
   | `calculate_social_interactions!(kinetics, social, states, social_buffers, neighbor_buffers, safe_eps)` | Compute the interaction and write it to `social_factor` |

### Notes

- Use `neighbor_buffers.weights[i, j]` for neighbor information (`> 0` means `j` is a neighbor of `i`)
- The existing models project the 2D force onto the normal of the heading and divide by `kinetics.v_0` to make it dimensionless:
  `social_buffers.social_factor[i] = dot(force, SVector(-s, c)) / kinetics.v_0`
- A model without interactions still needs an empty parameter struct and a buffer whose `social_factor` is set to zero

---

## Neighbor algorithm

Neighbors are fixed to Voronoi adjacency. `SimRunner.execute_simulation` creates `Neighbor_Voronoi.Neighbor_Voronoi()` and passes it to `run_simulation`. The adjacency graph is computed with `DelaunayTriangulation` (Delaunay edges coincide with Voronoi adjacency). Using a different neighbor method requires editing `SimRunner.jl`, and also the estimator side, which reconstructs Delaunay neighbors in `src/QMLE/estimator_modules/Neighbor/Delaunay.py`.

**Reference implementation**: `Physics/Neighbor/Voronoi.jl`

### What to define

1. **Parameter struct** `Neighbor_<Name> <: AbstractNeighborMethod` (may have no parameters)

2. **Buffer struct** `NeighborBuffer_<Name>{T} <: AbstractBuffer` (for example `NeighborBuffer_Voronoi`)
   - Typical field: `weights::Matrix{T}` (`num_agents × num_agents`)
   - `weights[i, j] > 0` means agent `j` is a neighbor of agent `i`

3. **Required functions**

   | Function | Role |
   |----------|------|
   | `allocate_neighbor_buffers(neighbor_method, num_agents, T)` | Create the buffer |
   | `update_neighbors!(neighbor_method, state, neighbor_buffer, rng)` | Overwrite the neighbor matrix at every step |

### Notes

- Social modules assume that `neighbor_buffers.weights` exists
- The Voronoi implementation normalizes each row to sum to 1 when there are three or more agents. With two agents it skips the triangulation and sets `weights[1, 2] = weights[2, 1] = 1`. Keep the social modules consistent with these weights
- Neighbor modules do not need `reset_buffers!` (`update_neighbors!` overwrites the buffer)

---

## Python-side steps (separate from Julia)

1. Add an estimator to the matching category in `src/QMLE/estimator_modules/` (`Kinetics/` or `Social/`) and register it with `@registry.register_kinetics("Dynamics_<Name>")` or `@registry.register_social("Soc_<Name>")` from `src/QMLE/registry.py`. Modules in these directories are imported automatically by `src/QMLE/builder.py`
2. In `src/experiment_configs/parameter_presets/setting_*.toml`, add the module name under `[modules]` and its parameters under `[parameters.<Category>.<module name>]`
3. Add the optimizer bounds and multistart initial ranges to `MODEL_RANGES` in `src/experiment_configs/estimation.py`
4. Run `scripts/01_simulation/00_prepare_toml.py` to regenerate the TOML files
5. Rebuild the sysimage with `./run_setup.sh julia` so that the new model is included in the precompile run
6. To include a new social model in the paper figures, add it to `ALL_SOCIAL_MODELS` in `scripts/04_figures/_common.py`

---

## Other implementation notes

### DelaunayTriangulation during the build

Precompiling `DelaunayTriangulation` is heavy and can fail when memory runs out (the sysimage build needs roughly 9 GB). In that case, allocate swap space before running the build:

```bash
sudo fallocate -l 8G /swapfile_large
sudo chmod 600 /swapfile_large
sudo mkswap /swapfile_large
sudo swapon /swapfile_large
```
