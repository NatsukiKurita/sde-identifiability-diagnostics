module Soc_TK2020
    using StaticArrays
    using LinearAlgebra

    import ..Types: AbstractSocial, AbstractState, AbstractBuffer,
            allocate_social_buffers, reset_buffers!, calculate_social_interactions!
    
    export Soc_TK2020

    struct Soc_TK2020{T<:AbstractFloat} <: AbstractSocial
        k_sep::T
        k_coh::T
        k_alg::T
    end

    struct Buffer_Soc_TK2020{T <: AbstractFloat} <: AbstractBuffer
        social_factor::Vector{T}
    end

    function allocate_social_buffers(social::Soc_TK2020, num_agents::Int, T::Type{<:AbstractFloat}) 
        social_buffers = Buffer_Soc_TK2020{T}(zeros(T, num_agents))
        return social_buffers
    end

    function reset_buffers!(buffers::Buffer_Soc_TK2020{T}) where T <: AbstractFloat
        fill!(buffers.social_factor, zero(T))
    end

    # --- 社会的相互作用の計算 ---
    function calculate_social_interactions!(
        kinetics,
        social::Soc_TK2020,
        states,
        social_buffers,
        neighbor_buffers,
        safe_eps::T,
    ) where T <: AbstractFloat

        num_agents = length(states.positions)
        k_sep = social.k_sep
        k_coh = social.k_coh
        k_alg = social.k_alg

        for i in 1:num_agents
            pos_i = states.positions[i]
            ori_i = states.orientations[i]

            sep_force = SVector{2, T}(0.0, 0.0)
            coh_force = SVector{2, T}(0.0, 0.0)
            alg_force = SVector{2, T}(0.0, 0.0)

            # 反発力の計算
            for j in 1:num_agents
                if i != j && neighbor_buffers.weights[i, j] > 0
                    pos_j = states.positions[j]
                    r_ij = pos_j - pos_i
                    d_ij = norm(r_ij) + safe_eps
                    sep_force -= k_sep * r_ij / d_ij * neighbor_buffers.weights[i, j]
                end
            end

            # 吸引力・整列力の計算
            center_of_mass_others = SVector{2, T}(0.0, 0.0)
            avg_velocity_others = SVector{2, T}(0.0, 0.0)

            regularization = sum(neighbor_buffers.weights[i, :]) + safe_eps
            for j in 1:num_agents
                if i != j && neighbor_buffers.weights[i, j] > 0
                    center_of_mass_others += states.positions[j] * neighbor_buffers.weights[i, j] / regularization
                    avg_velocity_others += states.velocities[j] * neighbor_buffers.weights[i, j] / regularization
                end
            end
            
            dist_to_com = norm(center_of_mass_others - states.positions[i]) + safe_eps
            coh_force = k_coh * (center_of_mass_others - states.positions[i]) / dist_to_com

            diff_to_avg_vel = avg_velocity_others - states.velocities[i]
            norm_diff_vel = norm(diff_to_avg_vel) + safe_eps
            alg_force = k_alg * diff_to_avg_vel / norm_diff_vel

            # 合計
            total_force = sep_force + coh_force + alg_force
            s, c = sincos(ori_i)
            projected_total_force = dot(total_force, SVector{2, T}(-s, c))
            social_buffers.social_factor[i] = projected_total_force
        end
    end
end


                    
                    

            
