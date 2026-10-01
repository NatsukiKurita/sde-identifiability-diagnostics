module Types
    export AbstractKinetics, AbstractSocial, AbstractWallAvoidance, AbstractNeighborMethod
    export AbstractState, AbstractBuffer
    export allocate_states, allocate_neighbor_buffers, allocate_social_buffers, allocate_wall_buffers
    export calculate_social_interactions!, calculate_wall_avoidance!
    export update_state_by_kinetics!, update_neighbors!, reset_buffers!
    export copy_state!

    # Abstract types for different components of the model
    abstract type AbstractKinetics end
    abstract type AbstractSocial end
    abstract type AbstractWallAvoidance end
    abstract type AbstractNeighborMethod end

    # Abstract types for states and buffers
    abstract type AbstractState end
    abstract type AbstractBuffer end

    # Function signatures for core operations
    function allocate_states end
    function allocate_neighbor_buffers end
    function allocate_social_buffers end
    function allocate_wall_buffers end
    function calculate_social_interactions! end
    function calculate_wall_avoidance! end
    function update_state_by_kinetics! end
    function update_neighbors! end
    function reset_buffers! end
    function copy_state! end

end # module Types