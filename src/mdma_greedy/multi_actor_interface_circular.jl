
# multi_actor_interface.jl
# Runs a 1-drone / 2-actor MPC simulation and produces a dual-view GIF.
#   - Actor 1 : circular trajectory  (primary target)
#   - Actor 2 : figure-8 trajectory  (secondary — appears in FPV naturally)
# tracking Actor 1.

ENV["GKSwstype"] = "100"

include("./MPC.jl")
include("./DroneVisualizationFPV.jl")

using .MPC
using .MPC.ActorMesh
using .MPC.ActorTrajectory
using .DroneVisualizationFPV

# ─── configuration ────────────────────────────────────────────────────────────
const NUM_STEPS       = 200
const PRIMARY_IDX     = 1        # actor the drone tracks
const OUTPUT_FILE     = "multi_actor_fpv_circular.gif"
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

# Actor 2 (Circle, starting far from origin)
const A2_RADIUS       = 4.5
const A2_ANG_VEL      = 2π / 25.0
const A2_START_ANGLE  = π
const A2_ORIGIN       = [4.0, 4.0, 0.0]
const OUTPUT_FILE     = "multi_actor_fpv_circular.gif"

# MPC parameters
const HORIZON         = 10
const TS              = 0.2
const FOLLOW_DIST     = 2.5
const AX_MAX          = 1.5
const AZ_MAX          = 0.5 * 9.81
const ALPHA_MAX       = π / 2
const DRONE_INIT      = [5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ─── build actor trajectories ─────────────────────────────────────────────────
println("="^70)
println("Multi-Actor Drone Simulation")
println("="^70)

println("\n[1/3] Building actor trajectories…")

mesh = build_actor_mesh(
    actor_width   = ACTOR_WIDTH,
    actor_depth   = ACTOR_DEPTH,
    actor_height  = ACTOR_HEIGHT
)

actor1_traj = circular_trajectory(
    mesh;
    num_steps        = NUM_STEPS,
    radius           = A1_RADIUS,
    start_angle      = A1_START_ANGLE,
    angular_velocity = A1_ANG_VEL,
    init_position    = A1_ORIGIN,
    actor_id         = 1
)

actor2_traj = circular_trajectory(
    mesh;
    num_steps        = NUM_STEPS,
    radius           = A2_RADIUS,
    start_angle      = A2_START_ANGLE,
    angular_velocity = A2_ANG_VEL,
    init_position    = A2_ORIGIN,
    actor_id         = 2
)

all_actor_trajs = [actor1_traj, actor2_traj]
println("  ✓ Actor 1 (circle r=3, start=0°)   : $(length(actor1_traj)) steps")
println("  ✓ Actor 2 (circle r=4.5, start=180°): $(length(actor2_traj)) steps")

# ─── MPC drone loop (tracks Actor 1) ─────────────────────────────────────────
println("\n[2/3] Running MPC simulation (tracking Actor $PRIMARY_IDX)…")

params = RobotParameters(
    HORIZON,
    TS,
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

    horizon_end  = min(step + params.N - 1, length(primary_traj))
    actor_horizon = primary_traj[step:horizon_end]

    pos_opt, vel_opt = SafeTrajectory(
        current_pos,
        [0.0, 0.0, 0.0, 0.0],
        params,
        actor_horizon
    )

    current_pos = RobotDynamics(current_pos, vel_opt[:, 1], params.Ts)
    push!(drone_trajectory, copy(current_pos))
end

println("\n  ✓ Drone trajectory : $(length(drone_trajectory)) steps")

# ─── animate ──────────────────────────────────────────────────────────────────
println("\n[3/3] Rendering animation…")

animate_multi_actor(
    all_actor_trajs,
    drone_trajectory;
    primary_actor_idx = PRIMARY_IDX,
    anim_file         = OUTPUT_FILE,
    fps               = FPS,
    xlims             = (-8.0,  8.0),
    ylims             = (-8.0,  8.0),
    zlims             = ( 0.0,  4.0),
    camera            = (25, 45),
    fpv_tilt          = -0.35,
    fpv_fov           =  1.2,
    fpv_view_size     =  0.68
)

println("\n" * "="^70)
println("✓ Done!  →  $OUTPUT_FILE")
println("="^70)
