# fpv_position_heatmap.jl
# Generates a heatmap GIF showing the Gaussian distribution of the actor's
# position inside the FPV frame over time.

ENV["GKSwstype"] = "100"

module MPC
    include("ActorMesh.jl")
    include("ActorTrajectory.jl")
    using .ActorMesh
    using .ActorTrajectory
end

include("DroneVisualizationFPV.jl")

using .MPC
using .MPC.ActorMesh
using .MPC.ActorTrajectory
using .DroneVisualizationFPV
using Plots
using LinearAlgebra

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  (Same as first_person_view_rot.jl)
# ─────────────────────────────────────────────────────────────

const ACTOR_WIDTH   = 0.5
const ACTOR_DEPTH   = 0.3
const ACTOR_HEIGHT  = 0.8

const ACTOR_STEPS   = 150
const ACTOR_RADIUS  = 3.0
const ACTOR_ANG_VEL = 2π / 15.0

const DRONE_BEHIND  = 2.0
const DRONE_ABOVE   = 1.5
const DRONE_SIDE    = 0.43

const FPV_TILT      = -0.35
const FPV_FOV       =  1.2
const FPV_VIEW_SIZE =  0.68

const OUTPUT_FILE   = "src/mdma_greedy/drone_experiments/actor_position_heatmap.gif\"
const FPS           = 12

# ─────────────────────────────────────────────────────────────
#  BUILD TRAJECTORIES
# ─────────────────────────────────────────────────────────────

mesh = build_actor_mesh(
    actor_width=ACTOR_WIDTH, actor_depth=ACTOR_DEPTH, actor_height=ACTOR_HEIGHT
)

actor_traj = circular_trajectory(
    mesh; num_steps=ACTOR_STEPS, radius=ACTOR_RADIUS, angular_velocity=ACTOR_ANG_VEL, actor_id=1
)

drone_traj = Vector{Vector{Float64}}()
for actor in actor_traj
    heading = actor.heading
    drone_x = actor.x - DRONE_BEHIND * cos(heading) + DRONE_SIDE * (-sin(heading))
    drone_y = actor.y - DRONE_BEHIND * sin(heading) + DRONE_SIDE * ( cos(heading))
    drone_z = actor.z + DRONE_ABOVE
    push!(drone_traj, [drone_x, drone_y, drone_z, 0.3*cos(heading), 0.3*sin(heading), 0.0, heading, 0.1])
end

# ─────────────────────────────────────────────────────────────
#  COLLECT FPV POINTS
# ─────────────────────────────────────────────────────────────
fpv_points = []
for i in 1:length(actor_traj)
    actor = actor_traj[i]
    drone = drone_traj[i]
    center_world = [actor.x, actor.y, actor.z + actor.mesh.height / 2.0]
    cp, ok = project_to_fpv(center_world, drone; tilt_angle=FPV_TILT, focal_length=FPV_FOV)
    if ok
        push!(fpv_points, cp)
    end
end
println("Collected $(length(fpv_points)) valid FPV projections.")

# ─────────────────────────────────────────────────────────────
#  HEATMAP GENERATION
# ─────────────────────────────────────────────────────────────

function draw_running_heatmap!(p, points_so_far; vs=0.68)
    u_vals = range(-vs, vs, length=80)
    v_vals = range(-vs*0.72, vs*0.72, length=80)
    
    # Gaussian kernel size for each point
    sigma = vs / 20.0 
    
    z_vals = zeros(length(v_vals), length(u_vals))
    for (i, v) in enumerate(v_vals)
        for (j, u) in enumerate(u_vals)
            val = 0.0
            for pt in points_so_far
                val += exp(-((u - pt[1])^2 + (v - pt[2])^2) / (2*sigma^2))
            end
            z_vals[i, j] = val
        end
    end

    contourf!(p, u_vals, v_vals, z_vals, 
              levels=15, color=:inferno, alpha=0.85, linewidth=0, colorbar=false)
end

println("Rendering Heatmap GIF...")
vs = FPV_VIEW_SIZE

actor_xs = [a.x for a in actor_traj]
actor_ys = [a.y for a in actor_traj]
actor_zs = [a.z for a in actor_traj]
drone_xs = [d[1] for d in drone_traj]
drone_ys = [d[2] for d in drone_traj]
drone_zs = [d[3] for d in drone_traj]

anim = @animate for i in 1:length(fpv_points)
    actor = actor_traj[i]
    drone = drone_traj[i]

    # --- World View ---
    p_world = plot(
        xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
        title="World View  |  Frame $i / $(length(fpv_points))",
        legend=:topright, camera=(30, 45),
        xlims=(-6.0, 6.0), ylims=(-6.0, 6.0), zlims=(0.0, 5.0),
        background_color=:white, size=(700, 700)
    )
    plot!(p_world, actor_xs[1:i], actor_ys[1:i], actor_zs[1:i], linewidth=3, color=:blue, alpha=0.6, label="Actor Path")
    plot!(p_world, drone_xs[1:i], drone_ys[1:i], drone_zs[1:i], linewidth=3, color=:red, alpha=0.6, label="Drone Path")
    world_verts = actor_world_vertices(actor.mesh, actor.x, actor.y, actor.z, actor.heading)
    draw_colored_actor!(p_world, actor, world_verts)
    draw_quadcopter!(p_world, drone, 0.3, 0.15)
    plot!(p_world, [drone[1], actor.x], [drone[2], actor.y], [drone[3], actor.z], linestyle=:dash, color=:gray, linewidth=1, alpha=0.4, label="LoS")

    # --- FPV Panel ---
    p_fpv = plot(
        title="Actor Position FPV Heatmap (Frame 1 to $i)\nRule of Thirds Offset Track", 
        legend=false,
        xlims=(-vs, vs), ylims=(-vs * 0.72, vs * 0.72),
        aspect_ratio=:equal, background_color=:black,
        grid=false, ticks=false, framestyle=:box, size=(700, 700)
    )

    # Draw Rot guides underneath
    line_col = RGBA(1.0, 1.0, 1.0, 0.4)
    for frac in (1/3, 2/3)
        xv = -vs + frac * (2*vs)
        yh = -vs*0.72 + frac * (1.44*vs)
        plot!(p_fpv, [xv, xv], [-vs*0.72, vs*0.72]; color=line_col, linewidth=1.0)
        plot!(p_fpv, [-vs, vs], [yh, yh]; color=line_col, linewidth=1.0)
    end

    # Sum of Gaussians for all points up to i
    draw_running_heatmap!(p_fpv, fpv_points[1:i]; vs=vs)
    
    # Highlight current position
    cp = fpv_points[i]
    scatter!(p_fpv, [cp[1]], [cp[2]], markersize=8, color=:white, markerstrokewidth=2, markerstrokecolor=:black, label="")

    plot(p_world, p_fpv, layout=(1, 2), size=(1400, 700))
end

println("Saving $(OUTPUT_FILE)...")
gif(anim, OUTPUT_FILE, fps=FPS)
println("Done!")
