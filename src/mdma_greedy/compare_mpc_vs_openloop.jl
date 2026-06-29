# compare_mpc_vs_openloop.jl
#
# Parallel simulation comparing two propagation strategies:
#
#   MPC (receding horizon):
#     At every step, solve SafeTrajectory → apply vel_opt[:,1] → RobotDynamics → next state
#
#   Open-loop (pos_opt direct):
#     At step 1, solve SafeTrajectory once → read pos_opt directly as the full trajectory
#     (no feedback, no re-solving)
#
# Produces a side-by-side 3D plot + a GIF showing both paths.

ENV["GKSwstype"] = "100"

include("./MPC.jl")

using .MPC
using .MPC.ActorTrajectory
using .MPC.ActorMesh
using Plots

# ── Simulation parameters (mirror multi_actor_interface.jl) ──────────────────

const NUM_STEPS   = 80
const HORIZON     = 10
const TS          = 0.2
const FOLLOW_DIST = 2.5
const AX_MAX      = 1.5
const AZ_MAX      = 0.5 * 9.81
const ALPHA_MAX   = π / 2
const DRONE_INIT  = [5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ── Build actor trajectory ────────────────────────────────────────────────────

mesh = build_actor_mesh(
    actor_width  = 0.5,
    actor_depth  = 0.3,
    actor_height = 0.8
)

actor_traj = circular_trajectory(
    mesh;
    num_steps        = NUM_STEPS,
    radius           = 3.0,
    start_angle      = 0.0,
    angular_velocity = 2π / 15.0,
    init_position    = [0.0, 0.0, 0.0],
    actor_id         = 1
)

params = RobotParameters(
    HORIZON,
    TS,
    [-AX_MAX, -AX_MAX, -AZ_MAX, -ALPHA_MAX],
    [ AX_MAX,  AX_MAX,  AZ_MAX,  ALPHA_MAX],
    FOLLOW_DIST,
    1.0   # safety_dist
)

println("="^60)
println("Parallel Simulation: MPC vs Open-Loop")
println("="^60)

# ── Simulation 1: MPC (receding horizon, uses vel_opt[:,1] each step) ─────────

println("\n[1/3] Running MPC simulation (receding horizon)…")

mpc_pos     = copy(DRONE_INIT)
mpc_traj    = [copy(mpc_pos)]

for step in 1:(length(actor_traj) - 1)
    global mpc_pos
    step % 20 == 0 && print("  • MPC step $step\r")
    h_end        = min(step + params.N - 1, length(actor_traj))
    actor_horizon = actor_traj[step:h_end]

    pos_opt, vel_opt = SafeTrajectory(mpc_pos, [0.0,0.0,0.0,0.0], params, actor_horizon)

    # Receding horizon: apply only the FIRST control, re-solve next step
    mpc_pos = RobotDynamics(mpc_pos, vel_opt[:, 1], params.Ts)
    push!(mpc_traj, copy(mpc_pos))
end
println("\n  ✓ MPC trajectory: $(length(mpc_traj)) steps")

# ── Simulation 2: Open-loop (solve ONCE at step 1, read pos_opt directly) ─────

println("\n[2/3] Running open-loop simulation (pos_opt direct)…")

# Solve once from the initial state using the first N steps of actor trajectory
h_end_ol      = min(1 + params.N - 1, length(actor_traj))
actor_horizon_ol = actor_traj[1:h_end_ol]

pos_opt_ol, _ = SafeTrajectory(DRONE_INIT, [0.0,0.0,0.0,0.0], params, actor_horizon_ol)

# pos_opt_ol is 8×(N+1) — each column is a planned state
openloop_traj = [pos_opt_ol[:, k] for k in 1:size(pos_opt_ol, 2)]
println("  ✓ Open-loop trajectory: $(length(openloop_traj)) steps (N+1 = $(params.N+1))")

# ── Extract x,y,z for plotting ────────────────────────────────────────────────

mpc_xs = [s[1] for s in mpc_traj]
mpc_ys = [s[2] for s in mpc_traj]
mpc_zs = [s[3] for s in mpc_traj]

ol_xs  = [s[1] for s in openloop_traj]
ol_ys  = [s[2] for s in openloop_traj]
ol_zs  = [s[3] for s in openloop_traj]

actor_xs = [a.x for a in actor_traj]
actor_ys = [a.y for a in actor_traj]
actor_zs = [a.z for a in actor_traj]

# ── Static side-by-side comparison plot ──────────────────────────────────────

println("\n[3/3] Rendering comparison plot and GIF…")

p_mpc = plot(
    mpc_xs, mpc_ys, mpc_zs,
    linewidth=3, color=:red, alpha=0.8, label="MPC (receding horizon)",
    xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
    title="MPC — Receding Horizon",
    xlims=(-8,8), ylims=(-8,8), zlims=(0,5),
    camera=(25,45), background_color=:white, legend=:topright
)
plot!(p_mpc, actor_xs, actor_ys, actor_zs,
      linewidth=2, color=:blue, alpha=0.5, label="Actor path")
scatter!(p_mpc, [mpc_xs[1]], [mpc_ys[1]], [mpc_zs[1]],
         markersize=6, color=:pink, marker=:square, label="Drone start")

p_ol = plot(
    ol_xs, ol_ys, ol_zs,
    linewidth=3, color=:darkorange, alpha=0.8, label="Open-loop (pos_opt, single solve)",
    xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
    title="Open-loop — pos_opt Direct",
    xlims=(-8,8), ylims=(-8,8), zlims=(0,5),
    camera=(25,45), background_color=:white, legend=:topright
)
# Show actor path only for the matching horizon length
n_ol = length(ol_xs)
plot!(p_ol, actor_xs[1:min(n_ol,end)], actor_ys[1:min(n_ol,end)], actor_zs[1:min(n_ol,end)],
      linewidth=2, color=:blue, alpha=0.5, label="Actor path (horizon)")
scatter!(p_ol, [ol_xs[1]], [ol_ys[1]], [ol_zs[1]],
         markersize=6, color=:pink, marker=:square, label="Drone start")

combined = plot(p_mpc, p_ol, layout=(1,2), size=(1400,700),
                plot_title="MPC (receding horizon) vs Open-loop (pos_opt)")
savefig(combined, "drone_experiments/mpc_vs_openloop_static.png")
println("  ✓ Static plot saved → drone_experiments/mpc_vs_openloop_static.png")

# ── Animated GIF: both paths growing frame-by-frame ──────────────────────────

num_frames = length(mpc_traj)

anim = @animate for i in 1:num_frames
    # Left: MPC path so far
    p1 = plot(
        mpc_xs[1:i], mpc_ys[1:i], mpc_zs[1:i],
        linewidth=3, color=:red, alpha=0.8, label="MPC",
        xlabel="X", ylabel="Y", zlabel="Z",
        title="MPC  (step $i)",
        xlims=(-8,8), ylims=(-8,8), zlims=(0,5),
        camera=(25,45), background_color=:white, legend=:topright
    )
    plot!(p1, actor_xs[1:i], actor_ys[1:i], actor_zs[1:i],
          linewidth=2, color=:blue, alpha=0.4, label="Actor")
    scatter!(p1, [mpc_xs[i]], [mpc_ys[i]], [mpc_zs[i]],
             markersize=7, color=:red, markerstrokewidth=1, label="")

    # Right: open-loop path (static — all N+1 points shown from frame 1)
    p2 = plot(
        ol_xs, ol_ys, ol_zs,
        linewidth=3, color=:darkorange, alpha=0.8, label="Open-loop (pos_opt)",
        xlabel="X", ylabel="Y", zlabel="Z",
        title="Open-loop pos_opt  (N=$(params.N) steps)",
        xlims=(-8,8), ylims=(-8,8), zlims=(0,5),
        camera=(25,45), background_color=:white, legend=:topright
    )
    plot!(p2, actor_xs[1:min(n_ol,end)], actor_ys[1:min(n_ol,end)], actor_zs[1:min(n_ol,end)],
          linewidth=2, color=:blue, alpha=0.4, label="Actor (horizon)")
    # Highlight where MPC currently is on the open-loop plan (if in range)
    if i <= n_ol
        scatter!(p2, [ol_xs[i]], [ol_ys[i]], [ol_zs[i]],
                 markersize=7, color=:darkorange, markerstrokewidth=1, label="")
    end

    plot(p1, p2, layout=(1,2), size=(1400,700))
end

gif(anim, "drone_experiments/mpc_vs_openloop.gif", fps=12)
println("  ✓ GIF saved → drone_experiments/mpc_vs_openloop.gif")

println("\n" * "="^60)
println("Done!")
println("  MPC steps    : $(length(mpc_traj))")
println("  Open-loop steps: $(length(openloop_traj))  (= N+1 = $(params.N+1))")
println("="^60)
