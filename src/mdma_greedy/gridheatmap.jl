using .ActorMesh
using .ActorTrajectory
using .DroneVisualizationFPV
using Plots
using LinearAlgebra

# ── ORIGINAL — reference-space dot-product formula ────────────────────────────
function isvisible_hard(dist::Vector{Float64}, face_normal::Vector{Float64})
    -dot(dist, face_normal) > 0.0 ? 1.0 : 0.0
end

function ppa_coverage_original(
    face::ActorFace,
    face_pos::Vector{Float64},    # world-frame face centre
    n_world::Vector{Float64},     # world-frame face normal
    drone_pos::Vector{Float64},
    drone_yaw::Float64
)
    alpha = 1
    dist    = face_pos .- drone_pos
    d4      = norm(dist)^4
    d4 < 1e-6 && return 0.0
    heading = [cos(drone_yaw), sin(drone_yaw), 0.0]

    pixel_density = alpha * abs(dot(dist, heading)) *
           (-dot(dist, n_world))  *
           isvisible_hard(dist, n_world) / d4

    return pixel_density
end

function ppa_quality_original(face::ActorFace, cov::Float64)
    ppa_original_value = face.weight * face.area * sqrt(max(cov, 0.0))
    return ppa_original_value
end





function animate_spatial_ppa_heatmap(
    actor_traj::Vector;
    drone_pos::Vector{Float64} = [4.0, 0.0, 2.0],
    drone_yaw::Float64 = pi,
    anim_file::String = "src/mdma_greedy/drone_experiments/PPA_SpatialHeatmap_Rotation.gif",
    fps::Int = 10,
    grid_size::Int = 61,
    x_range::Tuple{Float64, Float64} = (-0.70, 0.70),
    y_range::Tuple{Float64, Float64} = (-0.70, 0.70)
)

    preferred_face = actor.mesh.faces[1] # Front Face (:front)
    face_pos_pref  = actor_world_face_center(actor.mesh, preferred_face, actor.x, actor.y, actor.z, actor.heading)
    n_world_pref   = actor_world_normal(preferred_face, actor.heading)


    xs = range(x_range[1], x_range[2], length=grid_size)
    ys = range(y_range[1], y_range[2], length=grid_size)
    cmap = :turbo

    preferred_face = actor.mesh.faces[1] # Front Face (:front)

    drone = [drone_pos[1],drone_pos[2], 2.0, 0.0, 0.0, drone_yaw, 0.0]

    for (iy,y_a in enumerate(ys))
        for (ix,x_a in enumerate(xs))
            actor = ActorState(x_a, y_a, 0.0, heading, current_actor.mesh, 1)

            po = ppa_quality_original(preferred_face, ppa_coverage_original())
            H_orig[iy, ix]   = po
        end
    end



end

function run_heatmap(;num_steps=120)

#create a mesh

mesh = build_actor_mesh(
        actor_width=0.5, actor_depth=0.3, actor_height=0.8,
        front_weight=1.0, side_weight=0.5, top_weight=0.25,
        back_weight=0.2, bottom_weight=0.1
    )

    drone_fixed = [4.0, 0.0, 2.0, 0.0, 0.0, 0.0, pi, 0.0]
    drone_traj  = fill(drone_fixed, num_steps)
    actor_traj  = figure_eight_trajectory(mesh; num_steps=num_steps, scale=3.0, angular_velocity=2π/15.0, actor_id=1)

    # ── 1. 3D World View + FPV Camera Panel Animation (DroneVisualizationFPV.jl) ───
    println("\n[1/4] Rendering 3D World View + FPV Camera Panel Animation (Static Drone)...")
    out_fpv_world_gif = "src/mdma_greedy/drone_experiments/PPA_FPV_WorldView_Animation.gif"
    animate_drone_and_actor(actor_traj, drone_traj; anim_file=out_fpv_world_gif, fps=10)
    println("  Saved Animation GIF → $out_fpv_world_gif")

end 

run_heatmap()