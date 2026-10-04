import jax.numpy as jnp


def neighbor_delaunay_stepwise(params, states_t, aux_data_t):
    # Get the Delaunay adjacency from aux_data
    adjmat_t = aux_data_t["delaunay_neighbors"]  # shape: (N, E_max), -1 denotes a dummy edge
    N = states_t["positions"].shape[0]

    i_idx = jnp.arange(N)[:, None]  # shape: (N, 1)
    j_idx = adjmat_t  # shape: (N, E_max)

    # 1. First build an unweighted dense 0/1 adjacency matrix
    # Dummies (-1) are written to the last slot (index N)
    adj_pad = jnp.zeros((N + 1, N + 1))
    
    # Set 1.0 where an edge exists
    adj_pad = adj_pad.at[i_idx, j_idx].set(1.0) 

    # Drop the dummy slot to get the valid N x N adjacency matrix
    adj = adj_pad[:N, :N]

    # 2. Number of neighbors per agent (N_i) as row sums
    degrees = jnp.sum(adj, axis=1, keepdims=True)  # shape: (N, 1)

    # 3. Normalize each row by its neighbor count (avoiding division by zero)
    w_ij = jnp.where(degrees > 0.0, adj / degrees, 0.0)

    return w_ij