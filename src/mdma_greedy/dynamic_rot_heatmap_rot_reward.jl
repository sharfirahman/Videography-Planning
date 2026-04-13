# dynamic_rot_heatmap_rot_reward.jl
# Same as dynamic_rot_heatmap.jl but with the Rule-of-Thirds power-point reward
# enabled in the MPC (rot_weight > 0).  The drone is now optimised to keep
# the primary actor ON one of the four FPV power-point intersections.

ENV["GKSwstype"] = "100"

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
const PRIMARY_IDX     = 1
const FPS             = 12
const OUTPUT_FILE     = "dynamic_rot_heatmap_rot_reward.gif"

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

# ── Key difference: Rule-of-Thirds power-point reward weight ──────────────
const ROT_WEIGHT      = 5.0   # Tune: higher → stronger pull toward power points

# ─────────────────────────────────────────────────────────────
#  DYNAMIC HEATMAP LOGIC  (unchanged from dynamic_rot_heatmap.jl)
# ─────────────────────────────────────────────────────────────

function draw_dynamic_heatmap!(p, actor_u, actor_v, sec_u, sec_v; vs=0.68, plot_alpha=0.55)
    u_vals = range(-vs, vs, length=60)
    v_vals = range(-vs*0.72, vs*0.72, length=60)
    
    power_points = [
        (-vs/3, -vs*0.72/3),
        ( vs/3, -vs*0.72/3),
        (-vs/3,  vs*0.72/3),
        ( vs/3,  vs*0.72/3)
    ]
    
    activations = zeros(4)
    for (idx, (cu, cv)) in enumerate(power_points)
        activation = 0.15  # Baseline glow

        if actor_u !== nothing && actor_v !== nothing
            dist_sq = (actor_u - cu)^2 + (actor_v - cv)^2
            activation += 1.0 * exp(-dist_sq / (2 * (0.15)^2))
        end

        if sec_u !== nothing && sec_v !== nothing
            dist_sq2 = (sec_u - cu)^2 + (sec_v - cv)^2
            activation += 0.4 * exp(-dist_sq2 / (2 * (0.15)^2))
        end

        activations[idx] = min(1.15, activation)
    end
    
    z_vals = zeros(length(v_vals), length(u_vals))
    for (i, v) in enumerate(v_vals)
        for (j, u) in enumerate(u_vals)
            val = 0.0
            for (idx, (cu, cv)) in enumerate(power_points)
                val += activations[idx] * exp(-((u - cu)^2)/(2*(vs/4.5)^2) -
                                               ((v - cv)^2)/(2*(vs*0.72/4.5)^2))
            end
            z_vals[i, j] = val
        end
    end

    heatmap_grad = cgrad([:black, :darkred, :orange, :yellow, :white])
    contourf!(p, u_vals, v_vals, z_vals,
              levels=range(0, 1.15, length=20), color=heatmap_grad,
              alpha=plot_alpha, linewidth=0, colorbar=false)
end

# ─────────────────────────────────────────────────────────────
#  BUILD TRAJECTORIES & RUN MPC
# ─────────────────────────────────────────────────────────────
println("Building trajectories and running MPC (RoT Reward, weight=$(ROT_WEIGHT))...")

mesh = build_actor_mesh(actor_width=ACTOR_WIDTH, actor_depth=ACTOR_DEPTH,
                        actor_height=ACTOR_HEIGHT)

actor1_traj = circular_trajectory(mesh; num_steps=NUM_STEPS, radius=A1_RADIUS,
    start_angle=A1_START_ANGLE, angular_velocity=A1_ANG_VEL,
    init_position=A1_ORIGIN, actor_id=1)

actor2_traj = lissajous_trajectory(mesh; num_steps=NUM_STEPS,
    init_position=A2_ORIGIN, amplitude_x=A2_AMPL_X, amplitude_y=A2_AMPL_Y,
    omega_x=A2_OMEGA_X, omega_y=A2_OMEGA_Y, phase=A2_PHASE, actor_id=2)

all_actor_trajs = [actor1_traj, actor2_traj]
primary_traj    = all_actor_trajs[PRIMARY_IDX]

# RoT reward enabled here ↓
params = RobotParameters(
    HORIZON, TS,
    [-AX_MAX, -AX_MAX, -AZ_MAX, -ALPHA_MAX],
    [ AX_MAX,  AX_MAX,  AZ_MAX,  ALPHA_MAX],
    FOLLOW_DIST, 1.5,
    ROT_WEIGHT   # ← Rule-of-Thirds power-point weight
)

current_pos      = copy(DRONE_INIT)
drone_trajectory = [copy(current_pos)]

for step in 1:(length(primary_traj) - 1)
    global current_pos
    step % 20 == 0 && print("  • Step $step/$(length(primary_traj)-1)\r")
    horizon_end   = min(step + params.N - 1, length(primary_traj))
    actor_horizon = primary_traj[step:horizon_end]
    pos_opt, vel_opt = SafeTrajectory(current_pos, [0.0, 0.0, 0.0, 0.0], params, actor_horizon)
    current_pos = RobotDynamics(current_pos, vel_opt[:, 1], params.Ts)
    push!(drone_trajectory, copy(current_pos))
end
println("\n  ✓ Drone trajectory: $(length(drone_trajectory)) steps")

# ─────────────────────────────────────────────────────────────
#  RENDERING
# ─────────────────────────────────────────────────────────────
println("Rendering animation...")

vs         = FPV_VIEW_SIZE
num_frames = min(minimum(length.(all_actor_trajs)), length(drone_trajectory))

anim = @animate for i in 1:num_frames
    drone = drone_trajectory[i]
    current_actors = [traj[i] for traj in all_actor_trajs]
    primary_actor  = current_actors[PRIMARY_IDX]

    # --- World View ---
    p_world = plot(
        xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
        title="World View  |  Frame $i / $num_frames  [RoT reward ON]",
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
        col  = (a_idx == PRIMARY_IDX) ? :blue : :orange
        plot!(p_world, [a.x for a in traj][1:i],
                       [a.y for a in traj][1:i],
                       [a.z for a in traj][1:i],
              linewidth=(a_idx == PRIMARY_IDX ? 3 : 2),
              color=col, alpha=0.6,
              label=(a_idx == PRIMARY_IDX ? "Primary Track" : "Actor 2"))
        world_verts = actor_world_vertices(actor.mesh, actor.x, actor.y,
                                           actor.z, actor.heading)
        draw_colored_actor!(p_world, actor, world_verts)
    end
    plot!(p_world, [drone[1], primary_actor.x], [drone[2], primary_actor.y],
                   [drone[3], primary_actor.z],
          linestyle=:dash, color=:gray, linewidth=1, alpha=0.4, label="LoS")

    # --- FPV Panel ---
    p_fpv = plot(
        title="FPV Camera  [RoT reward]", legend=false,
        xlims=(-vs, vs), ylims=(-vs * 0.72, vs * 0.72),
        aspect_ratio=:equal, background_color=:black,
        grid=false, ticks=false, framestyle=:box, size=(700, 700)
    )
    draw_fpv_panel!(p_fpv, current_actors, PRIMARY_IDX, drone,
                    [a.x for a in primary_traj][1:i],
                    [a.y for a in primary_traj][1:i],
                    [a.z for a in primary_traj][1:i];
                    tilt_angle=FPV_TILT, focal_length=FPV_FOV)
    # RoT grid lines
    line_col = RGBA(1.0, 1.0, 1.0, 0.4)
    for frac in (1/3, 2/3)
        xv = -vs + frac * (2*vs);  yh = -vs*0.72 + frac * (1.44*vs)
        plot!(p_fpv, [xv, xv], [-vs*0.72, vs*0.72]; color=line_col, linewidth=1.0)
        plot!(p_fpv, [-vs, vs], [yh, yh];            color=line_col, linewidth=1.0)
    end
    # Power-point dots
    for (pu, pv) in [(-vs/3,-vs*0.72/3),(vs/3,-vs*0.72/3),(-vs/3,vs*0.72/3),(vs/3,vs*0.72/3)]
        scatter!(p_fpv, [pu], [pv]; markersize=5, color=RGBA(1,0.85,0.1,0.7),
                 markerstrokewidth=0, label="")
    end

    # --- Heatmap Panel ---
    p_heat = plot(
        title="Dynamic Activation Heatmap  [RoT reward ON]", legend=false,
        xlims=(-vs, vs), ylims=(-vs * 0.72, vs * 0.72),
        aspect_ratio=:equal, background_color=:black,
        grid=false, ticks=false, framestyle=:box, size=(700, 700)
    )
    center_world = [primary_actor.x, primary_actor.y,
                    primary_actor.z + primary_actor.mesh.height / 2.0]
    cp,  ok  = project_to_fpv(center_world, drone; tilt_angle=FPV_TILT, focal_length=FPV_FOV)
    actor_u, actor_v = ok ? (cp[1], cp[2]) : (nothing, nothing)

    sec_actor       = current_actors[2]
    sec_center      = [sec_actor.x, sec_actor.y, sec_actor.z + sec_actor.mesh.height/2.0]
    cp2, ok2        = project_to_fpv(sec_center, drone; tilt_angle=FPV_TILT, focal_length=FPV_FOV)
    sec_u, sec_v    = ok2 ? (cp2[1], cp2[2]) : (nothing, nothing)

    draw_dynamic_heatmap!(p_heat, actor_u, actor_v, sec_u, sec_v; vs=vs, plot_alpha=1.0)
    for frac in (1/3, 2/3)
        xv = -vs + frac*(2*vs);  yh = -vs*0.72 + frac*(1.44*vs)
        plot!(p_heat, [xv, xv], [-vs*0.72, vs*0.72]; color=line_col, linewidth=1.0)
        plot!(p_heat, [-vs, vs], [yh, yh];            color=line_col, linewidth=1.0)
    end
    ok && scatter!(p_heat, [cp[1]], [cp[2]]; markersize=7, color=:white,
                   markerstrokewidth=2, markerstrokecolor=:green, label="")

    plot(p_world, p_fpv, p_heat, layout=(1, 3), size=(2100, 700))
end

println("Saving $(OUTPUT_FILE)...")
gif(anim, OUTPUT_FILE, fps=FPS)
println("Done! Saved to $OUTPUT_FILE")
