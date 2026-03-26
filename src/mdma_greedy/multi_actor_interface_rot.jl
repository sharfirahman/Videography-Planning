# multi_actor_interface_rot.jl
# Runs a 1-drone / 2-actor MPC simulation and produces a dual-view GIF
# with a Rule of Thirds overlay in the FPV panel.

ENV["GKSwstype"] = "100"

include("./MPC.jl")
include("./DroneVisualizationFPV.jl")

using .MPC
using .MPC.ActorMesh
using .MPC.ActorTrajectory
using .DroneVisualizationFPV
using Plots
using LinearAlgebra

const NUM_STEPS       = 200
const PRIMARY_IDX     = 1        # actor the drone tracks
const OUTPUT_FILE     = "src/mdma_greedy/drone_experiments/multi_actor_fpv_rot.gif\"
const FPS             = 12

# Actor geometry (shared mesh for both)
const ACTOR_WIDTH     = 0.5
const ACTOR_DEPTH     = 0.3
const ACTOR_HEIGHT    = 0.8

# Actor 1 (circle, radius 3, centred at origin)
const A1_RADIUS       = 3.0
const A1_ANG_VEL      = 2π / 15.0
const A1_START_ANGLE  = 0.0
const A1_ORIGIN       = [0.0, 0.0, 0.0]

# Actor 2 (Lissajous cross-mix, starts from [-4, -4] region)
const A2_AMPL_X      = 4.5
const A2_AMPL_Y      = 4.0
const A2_OMEGA_X     = 2π / 15.0
const A2_OMEGA_Y     = 2π / 10.0
const A2_PHASE       = π / 2
const A2_ORIGIN      = [-4.0, -4.0, 0.0]

# MPC parameters
const HORIZON         = 10
const TS              = 0.2
const FOLLOW_DIST     = 2.5
const AX_MAX          = 1.5
const AZ_MAX          = 0.5 * 9.81
const ALPHA_MAX       = π / 2
const DRONE_INIT      = [5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ─────────────────────────────────────────────────────────────
#  RULE-OF-THIRDS OVERLAY
# ─────────────────────────────────────────────────────────────

function draw_rot_overlay!(p, actor, drone; tilt_angle=-0.35, focal_length=1.2)
    kw = (tilt_angle=tilt_angle, focal_length=focal_length)
    x_lo, x_hi = Plots.xlims(p)
    y_lo, y_hi = Plots.ylims(p)
    x_w = x_hi - x_lo
    y_h = y_hi - y_lo

    # Guide lines
    line_col = RGBA(1.0, 1.0, 1.0, 0.28)
    for frac in (1/3, 2/3)
        xv = x_lo + frac * x_w
        yh = y_lo + frac * y_h
        plot!(p, [xv, xv], [y_lo, y_hi]; color=line_col, linewidth=1.0, label="")
        plot!(p, [x_lo, x_hi], [yh, yh]; color=line_col, linewidth=1.0, label="")
    end

    # Power-point dots
    power_pts = [(x_lo + fx * x_w, y_lo + fy * y_h) for fx in (1/3, 2/3) for fy in (1/3, 2/3)]
    scatter!(p, [q[1] for q in power_pts], [q[2] for q in power_pts];
             markersize=4, color=RGBA(1.0, 1.0, 1.0, 0.55), markerstrokewidth=0, label="")

    # "1/3" badge
    annotate!(p, x_lo + 0.012*x_w, y_hi - 0.045*y_h, text("1/3", :left, RGBA(1.0, 1.0, 1.0, 0.50), 6))

    # Actor-centre gold ring
    center_world = [actor.x, actor.y, actor.z + actor.mesh.height / 2.0]
    cp, ok = project_to_fpv(center_world, drone; kw...)
    if ok
        scatter!(p, [cp[1]], [cp[2]]; markersize=9, color=RGBA(1.0, 0.85, 0.1, 0.0),
                 markerstrokewidth=2.0, markerstrokecolor=RGBA(1.0, 0.85, 0.1, 0.90),
                 marker=:circle, label="")
        annotate!(p, cp[1], cp[2] + 0.05, text("RoT", :center, RGBA(1.0, 0.85, 0.1, 0.80), 6))
    end
end

# ─────────────────────────────────────────────────────────────
#  BUILD ACTOR TRAJECTORIES
# ─────────────────────────────────────────────────────────────

println("="^70)
println("Multi-Actor Drone Simulation (with Rule of Thirds)")
println("="^70)

println("\n[1/3] Building actor trajectories…")

mesh = build_actor_mesh(
    actor_width   = ACTOR_WIDTH,
    actor_depth   = ACTOR_DEPTH,
    actor_height  = ACTOR_HEIGHT
)

actor1_traj = circular_trajectory(
    mesh; num_steps=NUM_STEPS, radius=A1_RADIUS, start_angle=A1_START_ANGLE,
    angular_velocity=A1_ANG_VEL, init_position=A1_ORIGIN, actor_id=1
)

actor2_traj = lissajous_trajectory(
    mesh; num_steps=NUM_STEPS, init_position=A2_ORIGIN, amplitude_x=A2_AMPL_X,
    amplitude_y=A2_AMPL_Y, omega_x=A2_OMEGA_X, omega_y=A2_OMEGA_Y, phase=A2_PHASE, actor_id=2
)

all_actor_trajs = [actor1_traj, actor2_traj]
println("  ✓ Actor 1 (circle r=3, start=0°) : $(length(actor1_traj)) steps")
println("  ✓ Actor 2 (Lissajous cross-mix)  : $(length(actor2_traj)) steps")

# ─────────────────────────────────────────────────────────────
#  MPC DRONE TRACKING
# ─────────────────────────────────────────────────────────────

println("\n[2/3] Running MPC simulation (tracking Actor $PRIMARY_IDX)…")

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
println("\n  ✓ Drone trajectory : $(length(drone_trajectory)) steps")

# ─────────────────────────────────────────────────────────────
#  RENDERING
# ─────────────────────────────────────────────────────────────

println("\n[3/3] Rendering animation…")

# Configuration for rendering
xlims         = (-8.0, 8.0)
ylims         = (-8.0, 8.0)
zlims         = (0.0,  4.0)
camera        = (25, 45)
fpv_tilt      = -0.35
fpv_fov       = 1.2
fpv_view_size = 0.68
vs            = fpv_view_size

num_actors = length(all_actor_trajs)
num_frames = min(minimum(length.(all_actor_trajs)), length(drone_trajectory))

primary_xs = [a.x for a in primary_traj]
primary_ys = [a.y for a in primary_traj]
primary_zs = [a.z for a in primary_traj]

drone_xs = [d[1] for d in drone_trajectory]
drone_ys = [d[2] for d in drone_trajectory]
drone_zs = [d[3] for d in drone_trajectory]

anim = @animate for i in 1:num_frames
    drone = drone_trajectory[i]
    current_actors = [traj[i] for traj in all_actor_trajs]
    primary_actor = current_actors[PRIMARY_IDX]

    # --- World View ---
    p_world = plot(
        xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
        title="World View  |  Frame $i / $num_frames",
        legend=:topright, camera=camera,
        xlims=xlims, ylims=ylims, zlims=zlims,
        background_color=:white, size=(700, 700)
    )

    plot!(p_world, drone_xs[1:i], drone_ys[1:i], drone_zs[1:i],
          linewidth=3, color=:red, alpha=0.6, label="Drone Path")
    draw_quadcopter!(p_world, drone, 0.3, 0.15)

    for (a_idx, actor) in enumerate(current_actors)
        traj = all_actor_trajs[a_idx]
        axs = [a.x for a in traj]
        ays = [a.y for a in traj]
        azs = [a.z for a in traj]
        col = (a_idx == PRIMARY_IDX) ? :blue : :orange
        plot!(p_world, axs[1:i], ays[1:i], azs[1:i],
              linewidth=(a_idx == PRIMARY_IDX ? 3 : 2), 
              color=col, alpha=0.6, label=(a_idx == PRIMARY_IDX ? "Primary Path" : "Actor Utils"))
        scatter!(p_world, [axs[1]], [ays[1]], [azs[1]],
                 markersize=5, color=col, marker=:square, label="")
        
        world_verts = actor_world_vertices(actor.mesh, actor.x, actor.y, actor.z, actor.heading)
        draw_colored_actor!(p_world, actor, world_verts)
    end

    plot!(p_world,
          [drone[1], primary_actor.x], [drone[2], primary_actor.y], [drone[3], primary_actor.z],
          linestyle=:dash, color=:gray, linewidth=1, alpha=0.4, label="LoS")
    scatter!(p_world, [drone_xs[1]], [drone_ys[1]], [drone_zs[1]],
             markersize=6, color=:pink, marker=:square, label="Drone Start")

    # --- FPV Panel ---
    p_fpv = plot(
        title="FPV Camera — Rule of Thirds", legend=false,
        xlims=(-vs, vs), ylims=(-vs * 0.72, vs * 0.72),
        aspect_ratio=:equal, background_color=:black,
        foreground_color_axis=:white, foreground_color_border=:black,
        grid=false, ticks=false, framestyle=:box, size=(700, 700)
    )

    draw_fpv_panel!(p_fpv, current_actors, PRIMARY_IDX, drone,
                    primary_xs[1:i], primary_ys[1:i], primary_zs[1:i];
                    tilt_angle=fpv_tilt, focal_length=fpv_fov)
                    
    # Inject RoT overlay over the existing actors
    draw_rot_overlay!(p_fpv, primary_actor, drone; tilt_angle=fpv_tilt, focal_length=fpv_fov)

    annotate!(p_fpv, vs - 0.02, -vs*0.72 + 0.04, text("FPV · DRONE CAM", :right, :black, 8))
    annotate!(p_fpv, vs - 0.02,  vs*0.72 - 0.04, text("● REC", :right, :red, 8))

    plot(p_world, p_fpv, layout=(1, 2), size=(1400, 700))
end

println("Saving animation → $OUTPUT_FILE …")
gif(anim, OUTPUT_FILE, fps=FPS)

println("\n" * "="^70)
println("✓ Done!  →  $OUTPUT_FILE")
println("="^70)
