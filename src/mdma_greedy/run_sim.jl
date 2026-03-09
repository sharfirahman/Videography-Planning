# ─────────────────────────────────────────────────────────────────────────────
#  run_simulation.jl
#
#  Standalone runner.  All three module files must be in the same directory.
#
#  Load order matters:
#    1. ActorMesh.jl       – geometry structs (no dependencies)
#    2. ActorTrajectory.jl – pose + motion    (uses ActorMesh types)
#    3. DroneVisualization – drawing          (uses both)
#
#  Usage:
#    julia run_simulation.jl
#  or from the REPL:
#    include("run_simulation.jl")
# ─────────────────────────────────────────────────────────────────────────────

include("ActorMesh.jl")
include("ActorTrajectory.jl")
include("DroneVisualizationCamera.jl")

using .ActorMesh: ActorMeshStruct, build_actor_mesh
using .ActorTrajectory: ActorState, circular_trajectory,
                        figure_eight_trajectory, general_trajectory
using .DroneVisualizationCamera: animate_drone_and_actor

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# ── Actor mesh dimensions ─────────────────────────────────────────────────────
const ACTOR_WIDTH   = 0.5    # Y  (side-to-side) in m
const ACTOR_DEPTH   = 0.3    # X  (front-to-back) in m
const ACTOR_HEIGHT  = 0.8    # Z  (top-to-bottom) in m

# ── Face visibility weights ───────────────────────────────────────────────────
const FRONT_WEIGHT  = 1.0
const SIDE_WEIGHT   = 0.5
const TOP_WEIGHT    = 0.25
const BACK_WEIGHT   = 0.2
const BOTTOM_WEIGHT = 0.1

# ── Trajectory ────────────────────────────────────────────────────────────────
const ACTOR_STEPS   = 150
const ACTOR_RADIUS  = 3.0
const ACTOR_ANG_VEL = 2π / 15.0

# ── Drone offset from actor ───────────────────────────────────────────────────
const DRONE_BEHIND  = 2.0    # metres behind actor (along actor heading)
const DRONE_ABOVE   = 1.5    # metres above actor

# ── FPV camera ────────────────────────────────────────────────────────────────
const FPV_TILT      = -20.0  # gimbal tilt in degrees (negative = nose-down)

# ── Output ────────────────────────────────────────────────────────────────────
const OUTPUT_FILE   = "drone_follows_actor_fpv.gif"
const FPS           = 12
const WORLD_XLIMS   = (-6.0,  6.0)
const WORLD_YLIMS   = (-6.0,  6.0)
const WORLD_ZLIMS   = ( 0.0,  5.0)
const WORLD_CAMERA  = (30, 45)

# ─────────────────────────────────────────────────────────────────────────────
#  BUILD MESH  (once — shared across all trajectory steps)
# ─────────────────────────────────────────────────────────────────────────────

mesh = build_actor_mesh(
    actor_width   = ACTOR_WIDTH,
    actor_depth   = ACTOR_DEPTH,
    actor_height  = ACTOR_HEIGHT,
    front_weight  = FRONT_WEIGHT,
    side_weight   = SIDE_WEIGHT,
    top_weight    = TOP_WEIGHT,
    back_weight   = BACK_WEIGHT,
    bottom_weight = BOTTOM_WEIGHT
)

println("Mesh built:")
println("  Dimensions : $(mesh.depth) m (D) × $(mesh.width) m (W) × $(mesh.height) m (H)")
println("  Vertices   : $(length(mesh.vertices))")
println("  Faces      : $(length(mesh.faces))")
println("  Edges      : $(length(mesh.edges))")
println()
println("  Face summary:")
for f in mesh.faces
    println("    :$(f.name)  weight=$(f.weight)  area=$(round(f.area, digits=4)) m²  corners=$(f.corner_indices)")
end
println()

# ─────────────────────────────────────────────────────────────────────────────
#  BUILD ACTOR TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

actor_traj = circular_trajectory(
    mesh;
    num_steps        = ACTOR_STEPS,
    radius           = ACTOR_RADIUS,
    angular_velocity = ACTOR_ANG_VEL,
    actor_id         = 1
)

println("Actor trajectory : $(length(actor_traj)) steps")

# ─────────────────────────────────────────────────────────────────────────────
#  BUILD DRONE TRAJECTORY
#  Drone state vector: [x, y, z, vx, vy, vz, theta, omega]
#
#  The drone orbits just outside the actor's circle at radius
#  (ACTOR_RADIUS + DRONE_BEHIND), on the same radial line as the actor.
#  It yaws each frame to always face the actor.
#
#  This gives genuine drone movement through the world while keeping
#  the actor in frame — the FPV shows the actor shifting as the
#  relative geometry changes each step.
# ─────────────────────────────────────────────────────────────────────────────

drone_traj = Vector{Vector{Float64}}()

for actor in actor_traj
    # Radial angle of actor from world origin
    orbit_angle = atan(actor.y, actor.x)

    # Drone sits on the same radial line, further out
    drone_r = ACTOR_RADIUS + DRONE_BEHIND
    drone_x = drone_r * cos(orbit_angle)
    drone_y = drone_r * sin(orbit_angle)
    drone_z = actor.z + DRONE_ABOVE

    # Yaw to face the actor
    drone_yaw = atan(actor.y - drone_y, actor.x - drone_x)

    # Velocity from finite difference — estimated from radial motion
    vx = -DRONE_BEHIND * sin(orbit_angle)
    vy =  DRONE_BEHIND * cos(orbit_angle)

    push!(drone_traj, [drone_x, drone_y, drone_z, vx, vy, 0.0, drone_yaw, 0.0])
end

println("Drone trajectory : $(length(drone_traj)) steps")
println()

# ─────────────────────────────────────────────────────────────────────────────
#  RENDER
# ─────────────────────────────────────────────────────────────────────────────

animate_drone_and_actor(
    actor_traj,
    drone_traj;
    anim_file    = OUTPUT_FILE,
    fps          = FPS,
    xlims        = WORLD_XLIMS,
    ylims        = WORLD_YLIMS,
    zlims        = WORLD_ZLIMS,
    world_camera = WORLD_CAMERA,
    fpv_tilt     = FPV_TILT
)

println("Done!  →  $OUTPUT_FILE")