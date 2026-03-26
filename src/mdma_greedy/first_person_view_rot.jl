# first_person_view_rot.jl
# Experiment: same settings as first_person_view.jl + Rule-of-Thirds overlay.
#
# Run from the project root with the videography_planning conda env:
#   conda run -n videography_planning julia src/mdma_greedy/first_person_view_rot.jl

ENV["GKSwstype"] = "100"

# ── Minimal "MPC" wrapper — includes only the mesh/trajectory sub-modules,
#    no JuMP/Ipopt.  DroneVisualizationFPV uses "..MPC.ActorMesh" so it needs
#    a parent module named MPC in scope.
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
#  CONFIGURATION  (identical to first_person_view.jl)
# ─────────────────────────────────────────────────────────────

const ACTOR_WIDTH   = 0.5
const ACTOR_DEPTH   = 0.3
const ACTOR_HEIGHT  = 0.8

const FRONT_WEIGHT  = 1.0
const SIDE_WEIGHT   = 0.5
const TOP_WEIGHT    = 0.25
const BACK_WEIGHT   = 0.2
const BOTTOM_WEIGHT = 0.1

const ACTOR_STEPS   = 150
const ACTOR_RADIUS  = 3.0
const ACTOR_ANG_VEL = 2π / 15.0

const DRONE_BEHIND  = 2.0
const DRONE_ABOVE   = 1.5
# Lateral offset (perpendicular to actor heading) that places the actor
# on the right 1/3 vertical line: u = focal_length * d / cx ≈ vs/3
# cx ≈ DRONE_BEHIND*cos(FPV_TILT) + 1.1*sin(|FPV_TILT|) ≈ 2.26
# d  = (vs/3) * cx / focal_length ≈ 0.227 * 2.26 / 1.2 ≈ 0.43 m
const DRONE_SIDE    = 0.43   # metres left of actor heading (moves actor right in FPV)

const FPV_TILT      = -0.35
const FPV_FOV       =  1.2
const FPV_VIEW_SIZE =  0.68

const OUTPUT_FILE   = "src/mdma_greedy/drone_experiments/drone_follows_actor_fpv_rot.gif\"
const FPS           = 12
const WORLD_XLIMS   = (-6.0,  6.0)
const WORLD_YLIMS   = (-6.0,  6.0)
const WORLD_ZLIMS   = ( 0.0,  5.0)
const WORLD_CAMERA  = (30, 45)

# ─────────────────────────────────────────────────────────────
#  RULE-OF-THIRDS OVERLAY
# ─────────────────────────────────────────────────────────────

"""
    draw_rot_overlay!(p, actor, drone; tilt_angle, focal_length)

Draws a rule-of-thirds grid on FPV panel `p`:
  - 2 vertical + 2 horizontal faint-white guide lines
  - 4 power-point dots at the intersections
  - Gold ring on the projected body centre of `actor`
"""
function draw_rot_overlay!(p, actor, drone;
                           tilt_angle::Float64   = FPV_TILT,
                           focal_length::Float64 = FPV_FOV)

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
    power_pts = [(x_lo + fx * x_w, y_lo + fy * y_h)
                 for fx in (1/3, 2/3) for fy in (1/3, 2/3)]
    scatter!(p, [q[1] for q in power_pts], [q[2] for q in power_pts];
             markersize=4, color=RGBA(1.0, 1.0, 1.0, 0.55),
             markerstrokewidth=0, label="")

    # "1/3" badge
    annotate!(p, x_lo + 0.012*x_w, y_hi - 0.045*y_h,
              text("1/3", :left, RGBA(1.0, 1.0, 1.0, 0.50), 6))

    # Actor-centre gold ring
    center_world = [actor.x, actor.y, actor.z + actor.mesh.height / 2.0]
    cp, ok = project_to_fpv(center_world, drone; kw...)
    if ok
        scatter!(p, [cp[1]], [cp[2]];
                 markersize=9,
                 color=RGBA(1.0, 0.85, 0.1, 0.0),
                 markerstrokewidth=2.0,
                 markerstrokecolor=RGBA(1.0, 0.85, 0.1, 0.90),
                 marker=:circle, label="")
        annotate!(p, cp[1], cp[2] + 0.05,
                  text("RoT", :center, RGBA(1.0, 0.85, 0.1, 0.80), 6))
    end
end

# ─────────────────────────────────────────────────────────────
#  BUILD MESH & TRAJECTORIES
# ─────────────────────────────────────────────────────────────

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

actor_traj = circular_trajectory(
    mesh;
    num_steps        = ACTOR_STEPS,
    radius           = ACTOR_RADIUS,
    angular_velocity = ACTOR_ANG_VEL,
    actor_id         = 1
)

println("Actor trajectory : $(length(actor_traj)) steps")

drone_traj = Vector{Vector{Float64}}()
for actor in actor_traj
    heading = actor.heading
    # Behind the actor along its heading
    drone_x = actor.x - DRONE_BEHIND * cos(heading)
    drone_y = actor.y - DRONE_BEHIND * sin(heading)
    drone_z = actor.z + DRONE_ABOVE
    # Lateral offset: perpendicular-left of heading = (-sin, cos)
    # Moves drone left → actor appears at right 1/3 line in FPV
    drone_x += DRONE_SIDE * (-sin(heading))
    drone_y += DRONE_SIDE * ( cos(heading))
    push!(drone_traj, [drone_x, drone_y, drone_z,
                       0.3*cos(heading), 0.3*sin(heading), 0.0, heading, 0.1])
end

println("Drone trajectory : $(length(drone_traj)) steps")
println()

# ─────────────────────────────────────────────────────────────
#  RENDER
# ─────────────────────────────────────────────────────────────

actor_xs = [a.x for a in actor_traj]
actor_ys = [a.y for a in actor_traj]
actor_zs = [a.z for a in actor_traj]
drone_xs = [d[1] for d in drone_traj]
drone_ys = [d[2] for d in drone_traj]
drone_zs = [d[3] for d in drone_traj]

num_frames = min(length(actor_traj), length(drone_traj))
vs         = FPV_VIEW_SIZE

anim = @animate for i in 1:num_frames
    actor = actor_traj[i]
    drone = drone_traj[i]

    world_verts = actor_world_vertices(
        actor.mesh, actor.x, actor.y, actor.z, actor.heading)

    # World view
    p_world = plot(
        xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
        title="World View  |  Frame $i / $num_frames",
        legend=:topright, camera=WORLD_CAMERA,
        xlims=WORLD_XLIMS, ylims=WORLD_YLIMS, zlims=WORLD_ZLIMS,
        background_color=:white, size=(700, 700)
    )
    plot!(p_world, actor_xs[1:i], actor_ys[1:i], actor_zs[1:i],
          linewidth=3, color=:blue, alpha=0.6, label="Actor Path")
    plot!(p_world, drone_xs[1:i], drone_ys[1:i], drone_zs[1:i],
          linewidth=3, color=:red,  alpha=0.6, label="Drone Path")
    draw_colored_actor!(p_world, actor, world_verts)
    draw_quadcopter!(p_world, drone, 0.3, 0.15)
    plot!(p_world, [drone[1], actor.x], [drone[2], actor.y], [drone[3], actor.z],
          linestyle=:dash, color=:gray, linewidth=1, alpha=0.4, label="LoS")
    scatter!(p_world, [actor_xs[1]], [actor_ys[1]], [actor_zs[1]],
             markersize=6, color=:lightblue, marker=:square, label="Actor Start")
    scatter!(p_world, [drone_xs[1]], [drone_ys[1]], [drone_zs[1]],
             markersize=6, color=:pink, marker=:square, label="Drone Start")

    # FPV panel
    p_fpv = plot(
        title="FPV Camera — Rule of Thirds",
        legend=false,
        xlims=(-vs, vs), ylims=(-vs * 0.72, vs * 0.72),
        aspect_ratio=:equal, background_color=:black,
        foreground_color_axis=:white, foreground_color_border=:black,
        grid=false, ticks=false, framestyle=:box, size=(700, 700)
    )

    draw_fpv_panel!(p_fpv, actor, drone,
                    actor_xs[1:i], actor_ys[1:i], actor_zs[1:i];
                    tilt_angle=FPV_TILT, focal_length=FPV_FOV)

    draw_rot_overlay!(p_fpv, actor, drone;
                      tilt_angle=FPV_TILT, focal_length=FPV_FOV)

    annotate!(p_fpv, vs - 0.02, -vs*0.72 + 0.04,
              text("FPV · DRONE CAM", :right, :black, 8))
    annotate!(p_fpv, vs - 0.02,  vs*0.72 - 0.04,
              text("● REC", :right, :red, 8))

    plot(p_world, p_fpv, layout=(1, 2), size=(1400, 700))
end

println("Saving animation → $OUTPUT_FILE …")
gif(anim, OUTPUT_FILE, fps=FPS)
println("Done!  →  $OUTPUT_FILE")
