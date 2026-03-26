# visualize_rot_weights.jl
# This script specifically isolates the FPV frame and visualizes the
# Rule of Thirds cost/reward landscape by dividing the frame into a pixel grid
# and calculating the Gaussian weight of each pixel.

ENV["GKSwstype"] = "100"
using Plots
using LinearAlgebra

# FPV Box specifications
const FPV_VIEW_SIZE = 0.68
const vs = FPV_VIEW_SIZE
const aspect_ratio_v = 0.72

# Grid resolution for calculating pixels
const RESOLUTION = 200

# Power point coordinates (1/3 and 2/3 locations)
const H_THIRDS = [-vs/3, vs/3]
const V_THIRDS = [-vs * aspect_ratio_v / 3, vs * aspect_ratio_v / 3]

function get_gaussian_weight(u, v)
    sigma_u = vs / 4.5
    sigma_v = (vs * aspect_ratio_v) / 4.5
    
    weight = 0.0
    # Sum of 4 Gaussians at the power points
    for cu in H_THIRDS
        for cv in V_THIRDS
            weight += exp(-((u - cu)^2)/(2*sigma_u^2) - ((v - cv)^2)/(2*sigma_v^2))
        end
    end
    return weight
end

println("Calculating FPV pixel weights...")

u_vals = range(-vs, vs, length=RESOLUTION)
v_vals = range(-vs * aspect_ratio_v, vs * aspect_ratio_v, length=RESOLUTION)

z_weights = zeros(length(v_vals), length(u_vals))

for (i, v) in enumerate(v_vals)
    for (j, u) in enumerate(u_vals)
        z_weights[i, j] = get_gaussian_weight(u, v)
    end
end

println("Rendering the plot...")

p = heatmap(u_vals, v_vals, z_weights, 
    color=:inferno, 
    title="FPV Screen - Rule of Thirds Gaussian Weights",
    aspect_ratio=:equal,
    xlims=(-vs, vs), ylims=(-vs * aspect_ratio_v, vs * aspect_ratio_v),
    colorbar_title="Gaussian Weight",
    framestyle=:box,
    size=(800, 600)
)

# Draw the Rule of Thirds physical grid lines over the heatmap
line_col = RGBA(1.0, 1.0, 1.0, 0.5)

# Vertical grid lines
plot!(p, [H_THIRDS[1], H_THIRDS[1]], [-vs*aspect_ratio_v, vs*aspect_ratio_v], color=line_col, linewidth=1.5, label="")
plot!(p, [H_THIRDS[2], H_THIRDS[2]], [-vs*aspect_ratio_v, vs*aspect_ratio_v], color=line_col, linewidth=1.5, label="")

# Horizontal grid lines
plot!(p, [-vs, vs], [V_THIRDS[1], V_THIRDS[1]], color=line_col, linewidth=1.5, label="")
plot!(p, [-vs, vs], [V_THIRDS[2], V_THIRDS[2]], color=line_col, linewidth=1.5, label="")

# Draw intersection dots perfectly on the power points
for cu in H_THIRDS
    for cv in V_THIRDS
        scatter!(p, [cu], [cv], markersize=5, color=:white, markerstrokewidth=0, label="")
    end
end

output_filename = "src/mdma_greedy/drone_experiments/fpv_rot_weights.png\"
println("Saving to $output_filename...")
savefig(p, output_filename)
println("Done!")
