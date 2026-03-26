# multi_actor_position_heatmap.jl
# Runs a 2-actor MPC simulation with a Rule-of-Thirds virtual target offset.
# Visualizes the actual dynamic tracking variance of the actor in the FPV frame 
# as an accumulating Gaussian heatmap KDE over time.

ENV["GKSwstype"] = "100"

# Note: We don't use the MPC module wrapper here because SafeTrajectory 
# and RobotDynamics actively solve using JuMP from the real MPC.jl
include("./MPC.jl")
include("./DroneVisualizationFPV.jl")

using .MPC
using .MPC.ActorMesh
using .MPC.ActorTrajectory
using .DroneVisualizationFPV
using Plots
using LinearAlgebra

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────

const NUM_STEPS       = 200
const PRIMARY_IDX     = 1        # actor the drone tracks
const OUTPUT_FILE     = "src/mdma_greedy/drone_experiments/multi_actor_position_heatmap.gif\"
const FPS             = 12

const ACTOR_WIDTH     = 0.5
const ACTOR_DEPTH     = 0.3
const ACTOR_HEIGHT    = 0.8

const A1_RADIUS       = 3.0
const A1_ANG_VEL      = 2π / 15.0
const A1_START_ANGLE  = 0.0
const A1_ORIGIN       = [0.0, 0.0, 0.0]

const A2_AMPL_X       = 4.5
const A2_AMPL_Y       = 4.0
const A2_OMEGA_X      = 2π / 15.0
const A2_OMEGA_Y      = 2π / 10.0
const A2_PHASE        = π / 2
const A2_ORIGIN       = [-4.0, -4.0, 0.0]

const HORIZON         = 10
const TS              = 0.2
const FOLLOW_DIST     = 2.5
const AX_MAX          = 1.5
const AZ_MAX          = 0.5 * 9.81
const ALPHA_MAX       = π / 2
const DRONE_INIT      = [5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]

const FPV_TILT        = -0.35
const FPV_FOV         =  1.2
const FPV_VIEW_SIZE   =  0.68

# ─────────────────────────────────────────────────────────────
#  BUILD ACTOR TRAJECTORIES
# ─────────────────────────────────────────────────────────────
println("="^70)
println("Multi-Actor MPC Heatmap (Rule of Thirds Offset)")
println("="^70)
println("\n[1/3] Building actor trajectories…")

mesh = build_actor_mesh(actor_width=ACTOR_WIDTH, actor_depth=ACTOR_DEPTH, actor_height=ACTOR_HEIGHT)

actor1_traj = circular_trajectory(
    mesh; num_steps=NUM_STEPS, radius=A1_RADIUS, start_angle=A1_START_ANGLE,
    angular_velocity=A1_ANG_VEL, init_position=A1_ORIGIN, actor_id=1
)

actor2_traj = lissajous_trajectory(
    mesh; num_steps=NUM_STEPS, init_position=A2_ORIGIN, amplitude_x=A2_AMPL_X,
    amplitude_y=A2_AMPL_Y, omega_x=A2_OMEGA_X, omega_y=A2_OMEGA_Y, phase=A2_PHASE, actor_id=2
)

all_actor_trajs = [actor1_traj, actor2_traj]
println("  ✓ Actor 1 & 2 paths built (length=$(length(actor1_traj))).")

# ─────────────────────────────────────────────────────────────
#  DYNAMIC MPC DRONE TRACKING (WITH VIRTUAL TARGET OFFSET)
# ─────────────────────────────────────────────────────────────
println("\n[2/3] Running MPC simulation (tracking Actor $PRIMARY_IDX off-center)…")

params = RobotParameters(
    HORIZON, TS,
    [-AX_MAX, -AX_MAX, -AZ_MAX, -ALPHA_MAX],
    [ AX_MAX,  AX_MAX,  AZ_MAX,  ALPHA_MAX],
    FOLLOW_DIST
)

primary_traj    = all_actor_trajs[PRIMARY_IDX]
current_pos     = copy(DRONE_INIT)
drone_trajectory = [copy(current_pos)]

for step in 1:(length(primary_traj) - 1)
    global current_pos
    step % 20 == 0 && print("  • Step $step/$(length(primary_traj)-1)\r")

    horizon_end   = min(step + params.N - 1, length(primary_traj))
    actor_horizon = primary_traj[step:horizon_end]

    pos_opt, vel_opt = SafeTrajectory(
        current_pos, [0.0, 0.0, 0.0, 0.0], params, actor_horizon
    )

    current_pos = RobotDynamics(current_pos, vel_opt[:, 1], params.Ts)
    push!(drone_trajectory, copy(current_pos))
end
println("\n  ✓ Drone trajectory : $(length(drone_trajectory)) steps generated.")

# ─────────────────────────────────────────────────────────────
#  COLLECT REAL FPV POSITIONS
# ─────────────────────────────────────────────────────────────
# Now that the MPC has flown the drone, we project the REAL primary actor 
# into the drone's FPV camera. Since the MPC tracking has variance/lag, 
# this point will jitter dynamically around the desired Rule of Thirds point.

fpv_points = []
num_frames = min(minimum(length.(all_actor_trajs)), length(drone_trajectory))

for i in 1:num_frames
    actor = all_actor_trajs[PRIMARY_IDX][i]
    drone = drone_trajectory[i]
    center_world = [actor.x, actor.y, actor.z + actor.mesh.height / 2.0]
    cp, ok = project_to_fpv(center_world, drone; tilt_angle=FPV_TILT, focal_length=FPV_FOV)
    if ok
        push!(fpv_points, cp)
    end
end
println("\n[3/3] Found $(length(fpv_points)) valid FPV frames. Rendering Heatmap GIF...")

# ─────────────────────────────────────────────────────────────
#  RENDERING
# ─────────────────────────────────────────────────────────────

function draw_running_heatmap!(p, points_so_far; vs=0.68)
    u_vals = range(-vs, vs, length=80)
    v_vals = range(-vs*0.72, vs*0.72, length=80)
    
    # Gaussian kernel size controls the smear of the heat signature
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

vs = FPV_VIEW_SIZE
anim = @animate for i in 1:length(fpv_points)
    drone = drone_trajectory[i]
    current_actors = [traj[i] for traj in all_actor_trajs]
    primary_actor = current_actors[PRIMARY_IDX]

    # --- World View ---
    p_world = plot(
        xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
        title="World View  |  Frame $i / $(length(fpv_points))",
        legend=:topright, camera=(25, 45),
        xlims=(-8.0, 8.0), ylims=(-8.0, 8.0), zlims=(0.0, 4.0),
        background_color=:white, size=(700, 700)
    )

    plot!(p_world, [d[1] for d in drone_trajectory][1:i], 
                   [d[2] for d in drone_trajectory][1:i], 
                   [d[3] for d in drone_trajectory][1:i], 
          linewidth=3, color=:red, alpha=0.6, label="Drone Path")
    draw_quadcopter!(p_world, drone, 0.3, 0.15)

    for (a_idx, actor) in enumerate(current_actors)
        traj = all_actor_trajs[a_idx]
        col = (a_idx == PRIMARY_IDX) ? :blue : :orange
        plot!(p_world, [a.x for a in traj][1:i], 
                       [a.y for a in traj][1:i], 
                       [a.z for a in traj][1:i], 
              linewidth=(a_idx == PRIMARY_IDX ? 3 : 2), 
              color=col, alpha=0.6, label=(a_idx == PRIMARY_IDX ? "Primary Track" : "Actor Utils"))
        
        world_verts = actor_world_vertices(actor.mesh, actor.x, actor.y, actor.z, actor.heading)
        draw_colored_actor!(p_world, actor, world_verts)
    end
    
    # Line of sight
    plot!(p_world,
          [drone[1], primary_actor.x], [drone[2], primary_actor.y], [drone[3], primary_actor.z],
          linestyle=:dash, color=:gray, linewidth=1, alpha=0.4, label="LoS")

    # --- FPV Panel ---
    p_fpv = plot(
        title="Dynamic MPC Variance Heatmap\n(Actual Subject Position via RoT Target)", 
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

    # Sum of Gaussians for all dynamically tracked points up to i
    draw_running_heatmap!(p_fpv, fpv_points[1:i]; vs=vs)
    
    # Highlight current position
    cp = fpv_points[i]
    scatter!(p_fpv, [cp[1]], [cp[2]], markersize=8, color=:white, markerstrokewidth=2, markerstrokecolor=:black, label="")

    plot(p_world, p_fpv, layout=(1, 2), size=(1400, 700))
end

println("Saving $(OUTPUT_FILE)...")
gif(anim, OUTPUT_FILE, fps=FPS)
println("\n" * "="^70)
println("✓ Done!  →  $OUTPUT_FILE")
println("="^70)
