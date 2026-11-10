#Contains function to get the grid locations for different artictic framing rules.


module ArtisticRules

using Plots

export get_rule_of_thirds_points, get_center_points, get_bottom_right_rot_point, get_top_right_rot_point, draw_dynamic_heatmap!

"""
    get_rule_of_thirds_points(vs, aspect_ratio)

Returns FPV coordinates for the four Rule of Thirds power points.
"""
function get_rule_of_thirds_points(vs::Float64, aspect_ratio::Float64=0.72)
    return [
        (-vs/3.0, -vs*aspect_ratio/3.0),
        ( vs/3.0, -vs*aspect_ratio/3.0),
        (-vs/3.0,  vs*aspect_ratio/3.0),
        ( vs/3.0,  vs*aspect_ratio/3.0)
    ]
end

"""
    get_center_points(vs, aspect_ratio)

Returns FPV coordinate for keeping the actor in the exact center.
"""
function get_center_points(vs::Float64, aspect_ratio::Float64=0.72)
    return [
        (0.0, 0.0)
    ]
end


"""
    draw_dynamic_heatmap!(p, actor_u, actor_v, sec_u, sec_v; power_points, vs, aspect_ratio, plot_alpha)

Generic heatmap drawing based on any set of target power points.
"""
function draw_dynamic_heatmap!(p, actor_u, actor_v, sec_u, sec_v; 
                            power_points::Vector{Tuple{Float64,Float64}},
                            vs::Float64=0.68, aspect_ratio::Float64=0.72, 
                            plot_alpha::Float64=0.55)
    u_vals = range(-vs, vs, length=60)
    v_vals = range(-vs*aspect_ratio, vs*aspect_ratio, length=60)
    
    sigma_u = vs / 4.5
    sigma_v = (vs * aspect_ratio) / 4.5
    
    activations = zeros(length(power_points))
    for (idx, (cu, cv)) in enumerate(power_points)
        activation = 0.15 # Baseline glow
        
        # Primary Actor Contribution (100% intensity)
        if actor_u !== nothing && actor_v !== nothing
            dist_sq = (actor_u - cu)^2 + (actor_v - cv)^2
            activation += 1.0 * exp(-dist_sq / (2 * (0.15)^2))
        end
        
        # Secondary Actor Contribution (40% lower-priority intensity)
        if sec_u !== nothing && sec_v !== nothing
            dist_sq2 = (sec_u - cu)^2 + (sec_v - cv)^2
            activation += 0.4 * exp(-dist_sq2 / (2 * (0.15)^2))
        end
        
        activations[idx] = min(1.15, activation) # Cap max brightness
    end
    
    z_vals = zeros(length(v_vals), length(u_vals))
    for (i, v) in enumerate(v_vals)
        for (j, u) in enumerate(u_vals)
            val = 0.0
            for (idx, (cu, cv)) in enumerate(power_points)
                val += activations[idx] * exp(-((u - cu)^2)/(2*sigma_u^2) - ((v - cv)^2)/(2*sigma_v^2))
            end
            z_vals[i, j] = val
        end
    end

    # Keep levels absolute so the colors don't shifting dynamically
    # Use a custom gradient starting with pure black
    heatmap_grad = cgrad([:black, :darkred, :orange, :yellow, :white])
    contourf!(p, u_vals, v_vals, z_vals, 
              levels=range(0, 1.15, length=20), color=heatmap_grad, alpha=plot_alpha, linewidth=0, colorbar=false)
end

end
