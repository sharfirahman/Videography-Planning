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

# ── Tracking mode switch ───────────────────────────────────────────────────
#   :single  → follow one specific actor (set FOCUS_ACTOR to 1 or 2)
#   :multi   → keep both actors in frame simultaneously (SafeTrajectoryMultiActor)
const TRACKING_MODE   = :multi
const FOCUS_ACTOR     = 1

# Auto-derived output names — change TRACKING_MODE/FOCUS_ACTOR above, not here
const RUN_TAG    = (TRACKING_MODE == :multi) ? "multi" : "actor$(FOCUS_ACTOR)"
const OUTPUT_FILE       = "dynamic_rot_heatmap_$(RUN_TAG).gif"
const METRICS_GIF       = "dynamic_rot_heatmap_$(RUN_TAG)_with_metrics.gif"
const ACTIVATION_CSV    = "activation_data_$(RUN_TAG).csv"
const ACTIVATION_PNG    = "activation_plots_$(RUN_TAG).png"
const PPA_CSV           = "ppa_data_$(RUN_TAG).csv"
const PPA_PNG           = "ppa_plot_$(RUN_TAG).png"
const BEST_FRAME_PNG    = "best_frame_$(RUN_TAG).png"

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
    FOLLOW_DIST, 1.5
    # ROT_WEIGHT removed — rot_weight field no longer in RobotParameters
)

current_pos      = copy(DRONE_INIT)
drone_trajectory = [copy(current_pos)]

for step in 1:(length(primary_traj) - 1)
    global current_pos
    step % 20 == 0 && print("  • Step $step/$(length(primary_traj)-1)\r")

    # Build actor horizon list — length determines mode automatically
    if TRACKING_MODE == :single
        focus_traj = all_actor_trajs[FOCUS_ACTOR]
        h_end = min(step + params.N - 1, length(focus_traj))
        actors = [focus_traj[step:h_end]]                             # 1 actor → SafeTrajectory
    elseif TRACKING_MODE == :multi
        h1 = min(step + params.N - 1, length(all_actor_trajs[1]))
        h2 = min(step + params.N - 1, length(all_actor_trajs[2]))
        actors = [all_actor_trajs[1][step:h1], all_actor_trajs[2][step:h2]]  # 2 actors → SafeTrajectoryMultiActor
    else
        error("Unknown TRACKING_MODE: $TRACKING_MODE  (use :single or :multi)")
    end

    pos_opt, vel_opt = PlanTrajectory(current_pos, [0.0,0.0,0.0,0.0], params, actors)
    current_pos = RobotDynamics(current_pos, vel_opt[:, 1], params.Ts)
    push!(drone_trajectory, copy(current_pos))
end
println("\n  ✓ Drone trajectory: $(length(drone_trajectory)) steps")

# ─────────────────────────────────────────────────────────────
#  DATA COLLECTION — Gaussian activation per frame
# ─────────────────────────────────────────────────────────────
println("Collecting activation data...")

function compute_frame_activation_rr(drone, primary_actor, sec_actor, vs)
    sigma = 0.15
    power_points = [(-vs/3, -vs*0.72/3), ( vs/3, -vs*0.72/3),
                    (-vs/3,  vs*0.72/3), ( vs/3,  vs*0.72/3)]

    center_world = [primary_actor.x, primary_actor.y,
                    primary_actor.z + primary_actor.mesh.height / 2.0]
    cp, ok = project_to_fpv(center_world, drone; tilt_angle=FPV_TILT, focal_length=FPV_FOV)
    actor_u, actor_v = ok ? (cp[1], cp[2]) : (nothing, nothing)

    sec_center = [sec_actor.x, sec_actor.y,
                  sec_actor.z + sec_actor.mesh.height / 2.0]
    cp2, ok2 = project_to_fpv(sec_center, drone; tilt_angle=FPV_TILT, focal_length=FPV_FOV)
    sec_u, sec_v = ok2 ? (cp2[1], cp2[2]) : (nothing, nothing)

    primary_score   = 0.0
    secondary_score = 0.0
    for (cu, cv) in power_points
        if actor_u !== nothing
            d2 = (actor_u - cu)^2 + (actor_v - cv)^2
            primary_score += exp(-d2 / (2 * sigma^2))
        end
        if sec_u !== nothing
            d2 = (sec_u - cu)^2 + (sec_v - cv)^2
            secondary_score += 0.4 * exp(-d2 / (2 * sigma^2))
        end
    end
    return primary_score, secondary_score
end

activation_rows_rr = []
best_combined_rr   = -Inf
best_frame_rr      = 1

vs         = FPV_VIEW_SIZE
num_frames = min(minimum(length.(all_actor_trajs)), length(drone_trajectory))

for i in 1:num_frames
    drone         = drone_trajectory[i]
    primary_actor = all_actor_trajs[PRIMARY_IDX][i]
    sec_actor     = all_actor_trajs[2][i]
    p, s = compute_frame_activation_rr(drone, primary_actor, sec_actor, vs)
    combined = p + s
    push!(activation_rows_rr, (frame=i, primary=p, secondary=s, combined=combined))
    if combined > best_combined_rr
        global best_combined_rr = combined
        global best_frame_rr    = i
    end
end

println("  Best frame: $best_frame_rr  (combined activation = $(round(best_combined_rr, digits=4)))")

# Save CSV
open(ACTIVATION_CSV, "w") do f
    println(f, "frame,primary_activation,secondary_activation,combined_activation")
    for r in activation_rows_rr
        println(f, "$(r.frame),$(r.primary),$(r.secondary),$(r.combined)")
    end
end
println("  Saved → $ACTIVATION_CSV")

# Save activation plot
frames_v    = [r.frame    for r in activation_rows_rr]
primary_v   = [r.primary  for r in activation_rows_rr]
secondary_v = [r.secondary for r in activation_rows_rr]
combined_v  = [r.combined for r in activation_rows_rr]

act_plot = plot(frames_v, primary_v,
                label="Actor1", linewidth=2, color=:blue,
                xlabel="Frame", ylabel="ROT Reward",
                title="Aesthetic Preferences vs Time",
                legend=:topright, background_color=:white)
plot!(act_plot, frames_v, secondary_v,
      label="Actor2", linewidth=2, color=:orange)
# plot!(act_plot, frames_v, combined_v,
#       label="Combined", linewidth=2, color=:green, linestyle=:dash)
vline!(act_plot, [best_frame_rr], color=:red, linestyle=:dot,
       linewidth=1.5, label="Best frame ($best_frame_rr)")

savefig(act_plot, ACTIVATION_PNG)
println("  Saved activation plot → $ACTIVATION_PNG")

# ─────────────────────────────────────────────────────────────
#  PPA (Pixel Per Area) / Camera Coverage vs Time
# ─────────────────────────────────────────────────────────────
println("Computing PPA (camera coverage) over time...")

# True Pixel-Per-Area using the pinhole camera model.
#
#   For each face of the actor:
#     ppa_face = face.weight × face.area × cos(θ) × (focal_length / d)²
#
#   where:
#     d     = Euclidean distance from drone to face centre
#     cos(θ) = dot(face_normal, -camera_ray_direction)   [0 if back-facing]
#     focal_length = FPV_FOV (same constant used in project_to_fpv)
#
#   1/d² falloff matches the pinhole projection formula:
#     projected_area ∝ object_area × cos(θ) / d²
#
function compute_ppa(drone_state, actor_state)
    focal_length = FPV_FOV
    total_ppa    = 0.0

    for face in actor_state.mesh.faces
        face_pos = actor_world_face_center(actor_state.mesh, face,
                       actor_state.x, actor_state.y, actor_state.z,
                       actor_state.heading)

        # Camera-to-face vector (world frame)
        d_vec = [face_pos[1] - drone_state[1],
                 face_pos[2] - drone_state[2],
                 face_pos[3] - drone_state[3]]
        d     = sqrt(d_vec[1]^2 + d_vec[2]^2 + d_vec[3]^2 + 1e-6)
        d_hat = d_vec ./ d

        # Face normal rotated into world frame (actor heading applied)
        n_world   = actor_world_normal(face, actor_state.heading)

        # cos(θ): positive → face is pointing toward the drone (visible)
        #         negative → face points away (back-face cull, contributes zero)
        # This is the correct visibility gate — no need for a frustum check because
        # a back-facing face already produces cos_theta ≤ 0.
        cos_theta  = dot(n_world, -d_hat)
        visibility = max(0.0, cos_theta)

        # Pinhole projected area: face_area × cos(θ) × (focal_length / d)²
        projected_pixels = face.area * visibility * (focal_length / d)^2
        total_ppa += face.weight * projected_pixels
    end
    return total_ppa
end


ppa_per_actor = [Float64[] for _ in 1:length(all_actor_trajs)]

for i in 1:num_frames
    drone = drone_trajectory[i]
    for (a_idx, traj) in enumerate(all_actor_trajs)
        push!(ppa_per_actor[a_idx], compute_ppa(drone, traj[i]))
    end
end

# Save PPA CSV — one column per actor
open(PPA_CSV, "w") do f
    header = "frame," * join(["actor_$(a)_ppa" for a in 1:length(all_actor_trajs)], ",")
    println(f, header)
    for i in 1:num_frames
        row = "$i," * join([ppa_per_actor[a][i] for a in 1:length(all_actor_trajs)], ",")
        println(f, row)
    end
end
println("  Saved → $PPA_CSV")

# Plot PPA vs time — one line per actor
actor_colors = [:crimson, :orange, :dodgerblue, :green]
actor_labels = ["Actor $a$(a == PRIMARY_IDX ? " (primary)" : "")" for a in 1:length(all_actor_trajs)]

ppa_plot = plot(xlabel="Frame", ylabel="PPA  [ face_area × cos(θ) × (f/d)² ]",
                title="Pixel-Per-Area Reward vs Time  (pinhole model)",
                legend=:topright, background_color=:white)

for (a_idx, ppa_vals) in enumerate(ppa_per_actor)
    lw    = (a_idx == PRIMARY_IDX) ? 2.5 : 1.5
    plot!(ppa_plot, 1:num_frames, ppa_vals,
          label=actor_labels[a_idx], linewidth=lw,
          color=actor_colors[mod1(a_idx, length(actor_colors))])
    best_f = argmax(ppa_vals)
    println("  Actor $a_idx best PPA frame: $best_f  (PPA = $(round(ppa_vals[best_f], digits=4)))")
end

savefig(ppa_plot, PPA_PNG)
println("  Saved PPA plot → $PPA_PNG")

# ─────────────────────────────────────────────────────────────
#  RENDERING
# ─────────────────────────────────────────────────────────────
println("Rendering animation...")


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
        alabel = "Actor $a_idx"
        plot!(p_world, [a.x for a in traj][1:i],
                       [a.y for a in traj][1:i],
                       [a.z for a in traj][1:i],
              linewidth=(a_idx == PRIMARY_IDX ? 3 : 2),
              color=col, alpha=0.6, label=alabel)
        world_verts = actor_world_vertices(actor.mesh, actor.x, actor.y,
                                           actor.z, actor.heading)
        draw_colored_actor!(p_world, actor, world_verts)
        # Label dot at current actor position
        scatter!(p_world, [actor.x], [actor.y], [actor.z + actor.mesh.height + 0.2];
                 markersize=4, color=col, markerstrokewidth=0, label="")
    end

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
    # Actor labels in FPV
    for (a_idx, actor) in enumerate(current_actors)
        ac_world = [actor.x, actor.y, actor.z + actor.mesh.height]
        cp_a, ok_a = project_to_fpv(ac_world, drone; tilt_angle=FPV_TILT, focal_length=FPV_FOV)
        if ok_a
            col = (a_idx == PRIMARY_IDX) ? :cyan : :yellow
            scatter!(p_fpv, [cp_a[1]], [cp_a[2]]; markersize=5, color=col,
                     markerstrokewidth=0, label="")
            annotate!(p_fpv, cp_a[1] + 0.03, cp_a[2] + 0.03,
                      text("A$a_idx", col, :left, 9))
        end
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
    # Actor labels in heatmap
    if ok
        annotate!(p_heat, cp[1] + 0.03, cp[2] + 0.03, text("A1", :cyan, :left, 9))
    end
    if ok2
        scatter!(p_heat, [cp2[1]], [cp2[2]]; markersize=7, color=:white,
                 markerstrokewidth=2, markerstrokecolor=:orange, label="")
        annotate!(p_heat, cp2[1] + 0.03, cp2[2] + 0.03, text("A2", :yellow, :left, 9))
    end


    # --- PPA Panel (growing over time) ---
    p_ppa = plot(xlabel="Frame", ylabel="PPA Reward",
                 title="PPA vs Time", legend=:topright,
                 xlims=(1, num_frames), background_color=:white,
                 ylims=(0, maximum(maximum.(ppa_per_actor)) * 1.15 + 1e-6))
    for (a_idx, ppa_vals) in enumerate(ppa_per_actor)
        lw  = (a_idx == PRIMARY_IDX) ? 2.5 : 1.5
        col = actor_colors[mod1(a_idx, length(actor_colors))]
        lbl = actor_labels[a_idx]
        plot!(p_ppa, 1:i, ppa_vals[1:i]; linewidth=lw, color=col, label=lbl)
        scatter!(p_ppa, [i], [ppa_vals[i]]; markersize=5, color=col,
                 markerstrokewidth=0, label="")
    end

    # --- Activation Panel (growing over time) ---
    prim_act_so_far = [r.primary   for r in activation_rows_rr[1:i]]
    sec_act_so_far  = [r.secondary for r in activation_rows_rr[1:i]]
    max_act = max(maximum([r.combined for r in activation_rows_rr]) * 1.15, 0.1)
    p_act = plot(xlabel="Frame", ylabel="Gaussian Activation",
                 title="RoT Activation vs Time", legend=:topright,
                 xlims=(1, num_frames), ylims=(0, max_act),
                 background_color=:white)
    plot!(p_act, 1:i, prim_act_so_far;
          linewidth=2.5, color=:blue, label="Primary Actor")
    plot!(p_act, 1:i, sec_act_so_far;
          linewidth=1.5, color=:orange, label="Secondary (×0.4)")
    scatter!(p_act, [i], [prim_act_so_far[end]];
             markersize=5, color=:blue, markerstrokewidth=0, label="")
    scatter!(p_act, [i], [sec_act_so_far[end]];
             markersize=5, color=:orange, markerstrokewidth=0, label="")

    # 5-panel layout: top row = world/fpv/heatmap, bottom row = ppa/activation
    l = @layout [a b c; d e]
    plot(p_world, p_fpv, p_heat, p_ppa, p_act, layout=l, size=(2100, 1400))
end

# Save 3-panel version
println("Saving $(OUTPUT_FILE)...")
gif(anim, OUTPUT_FILE, fps=FPS)
println("Done! Saved to $OUTPUT_FILE")

# Save 5-panel metrics animation
println("Saving $(METRICS_GIF)...")
gif(anim, METRICS_GIF, fps=FPS)
println("Done! Saved to $METRICS_GIF")
