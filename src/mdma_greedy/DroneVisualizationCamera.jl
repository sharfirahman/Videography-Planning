module DroneVisualizationCamera

using Plots
using LinearAlgebra
using ..ActorMesh: actor_world_vertices

export draw_quadcopter!, animate_drone_and_actor

# ─────────────────────────────────────────────────────────────────────────────
#  EDGE GEOMETRY
#
#  Converts an ActorMesh into the edge format Plots.jl expects:
#    x = [[x_start, x_end], ...]   one pair per edge
#    y = [[y_start, y_end], ...]
#    z = [[z_start, z_end], ...]
#
#  plot(x, y, z) draws each pair as a separate line segment — same pattern
#  as the unit cube example, applied to world-frame vertices.
# ─────────────────────────────────────────────────────────────────────────────

function actor_edge_coords(actor)
    wv = actor_world_vertices(actor.mesh, actor.x, actor.y, actor.z, actor.heading)
    x  = [[wv[i1][1], wv[i2][1]] for (i1, i2) in actor.mesh.edges]
    y  = [[wv[i1][2], wv[i2][2]] for (i1, i2) in actor.mesh.edges]
    z  = [[wv[i1][3], wv[i2][3]] for (i1, i2) in actor.mesh.edges]
    return x, y, z
end

# ─────────────────────────────────────────────────────────────────────────────
#  DRONE CAMERA ANGLES
#
#  Converts drone state [x,y,z,vx,vy,vz,yaw,omega] into (azimuth, elevation)
#  degrees for Plots.jl camera= kwarg used in the FPV panel.
#
#  azimuth  = drone yaw in degrees  (which way the drone is pointing)
#  elevation = fixed gimbal tilt    (negative = camera looks downward)
# ─────────────────────────────────────────────────────────────────────────────

function fpv_camera_angles(drone; tilt_deg=-20.0)
    azimuth = rad2deg(drone[7])
    return (azimuth, tilt_deg)
end

# ─────────────────────────────────────────────────────────────────────────────
#  LOCAL FRAME TRANSFORM
#
#  Translates and rotates world-frame coordinates into the drone's local frame
#  so the drone is always at the origin looking along +X.
#
#  Steps:
#    1. Subtract drone position  → drone is at (0,0,0)
#    2. Rotate by -yaw           → drone always faces +X
#
#  After this transform, camera=(0,0) in Plots.jl gives a true FPV —
#  looking straight ahead along +X from the origin.
# ─────────────────────────────────────────────────────────────────────────────

function to_drone_frame(xs, ys, zs, drone)
    dx, dy = drone[1], drone[2]
    dz     = drone[3]
    yaw    = drone[7]
    c, s   = cos(-yaw), sin(-yaw)

    # Translate then rotate each point
    xs_l = [c*(x - dx) - s*(y - dy) for (x, y) in zip(xs, ys)]
    ys_l = [s*(x - dx) + c*(y - dy) for (x, y) in zip(xs, ys)]
    zs_l = [z - dz                   for z       in zs         ]
    return xs_l, ys_l, zs_l
end

# Same transform for edges — each edge is a [start, end] pair
function edges_to_drone_frame(ex, ey, ez, drone)
    dx, dy = drone[1], drone[2]
    dz     = drone[3]
    yaw    = drone[7]
    c, s   = cos(-yaw), sin(-yaw)

    ex_l = [[c*(x - dx) - s*(y - dy) for (x,y) in zip(ex[k], ey[k])] for k in eachindex(ex)]
    ey_l = [[s*(x - dx) + c*(y - dy) for (x,y) in zip(ex[k], ey[k])] for k in eachindex(ex)]
    ez_l = [[z - dz                   for z       in ez[k]           ] for k in eachindex(ez)]
    return ex_l, ey_l, ez_l
end

# ─────────────────────────────────────────────────────────────────────────────
#  QUADCOPTER MODEL
# ─────────────────────────────────────────────────────────────────────────────

function draw_quadcopter!(p, drone_state, arm_length=0.3, prop_radius=0.15)
    x, y, z, vx, vy, vz, theta, omega = drone_state

    scatter!(p, [x], [y], [z],
             markersize=8, color=:darkred,
             markerstrokewidth=2, markerstrokecolor=:black, label="")

    for (i, arm_offset) in enumerate([π/4, 3π/4, 5π/4, 7π/4])
        angle = theta + arm_offset
        arm_x = x + arm_length * cos(angle)
        arm_y = y + arm_length * sin(angle)

        plot!(p, [x, arm_x], [y, arm_y], [z, z],
              color=:black, linewidth=3, label="")
        scatter!(p, [arm_x], [arm_y], [z],
                 markersize=4, color=:gray,
                 markerstrokewidth=1, markerstrokecolor=:black, label="")

        prop_θ = range(0, 2π; length=20)
        prop_x = arm_x .+ prop_radius .* cos.(prop_θ) .* cos(angle) .-
                           prop_radius .* sin.(prop_θ) .* sin(angle)
        prop_y = arm_y .+ prop_radius .* cos.(prop_θ) .* sin(angle) .+
                           prop_radius .* sin.(prop_θ) .* cos(angle)
        plot!(p, prop_x, prop_y, fill(z + 0.05, length(prop_θ)),
              color=(i % 2 == 0 ? :red : :blue), linewidth=2, alpha=0.6, label="")
    end

    cam_x = x + 0.15 * cos(theta)
    cam_y = y + 0.15 * sin(theta)
    plot!(p, [x, cam_x], [y, cam_y], [z, z - 0.1],
          color=:lime, linewidth=3, label="")
    scatter!(p, [cam_x], [cam_y], [z - 0.1],
             markersize=3, color=:lime, marker=:square, label="")
end

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ANIMATION  –  World view (ortho) LEFT  |  FPV (persp) RIGHT
#
#  Both panels plot the exact same world-frame edge coordinates.
#  The only difference between them is:
#    World view → proj_type=:ortho,  camera = fixed isometric angle
#    FPV        → proj_type=:persp,  camera = (drone_yaw, gimbal_tilt)
#
#  Plots.jl handles all the projection math — no manual math needed.
# ─────────────────────────────────────────────────────────────────────────────

"""
    animate_drone_and_actor(actor_trajectory, drone_trajectory; kwargs...)

Side-by-side animation:
  • **Left**  – orthographic isometric world view
  • **Right** – perspective FPV from drone camera

Both panels share the same world-frame geometry from ActorMesh.
The FPV uses Plots.jl's native proj_type=:persp with camera angles
derived from the drone's yaw and gimbal tilt each frame.

# Keyword arguments
| Argument       | Default                      | Description                        |
|----------------|------------------------------|------------------------------------|
| `anim_file`    | `"drone_actor_tracking.gif"` | Output GIF filename                |
| `fps`          | `10`                         | Frames per second                  |
| `xlims`        | `(-6, 6)`                    | World axis X limits                |
| `ylims`        | `(-6, 6)`                    | World axis Y limits                |
| `zlims`        | `(0, 5)`                     | World axis Z limits                |
| `world_camera` | `(45, 35)`                   | Isometric camera (azimuth°, elev°) |
| `fpv_tilt`     | `-20.0`                      | Gimbal tilt degrees (neg = down)   |
"""
function animate_drone_and_actor(
    actor_trajectory::Vector,
    drone_trajectory::Vector{Vector{Float64}};
    anim_file::String   = "drone_actor_tracking.gif",
    fps::Int            = 10,
    xlims::Tuple        = (-6.0,  6.0),
    ylims::Tuple        = (-6.0,  6.0),
    zlims::Tuple        = ( 0.0,  5.0),
    world_camera::Tuple = (45, 35),
    fpv_tilt::Float64   = -20.0
)
    actor_xs = [a.x for a in actor_trajectory]
    actor_ys = [a.y for a in actor_trajectory]
    actor_zs = [a.z for a in actor_trajectory]

    drone_xs = [d[1] for d in drone_trajectory]
    drone_ys = [d[2] for d in drone_trajectory]
    drone_zs = [d[3] for d in drone_trajectory]

    num_frames = min(length(actor_trajectory), length(drone_trajectory))

    # Shared axis kwargs — same world, same limits, same labels in both panels
    axis_kw = (
        xlabel       = "X (m)",
        ylabel       = "Y (m)",
        zlabel       = "Z (m)",
        xlims        = xlims,
        ylims        = ylims,
        zlims        = zlims,
        aspect_ratio = :equal,
        grid         = true,
        label        = :none,
    )

    anim = @animate for i in 1:num_frames
        actor    = actor_trajectory[i]
        drone    = drone_trajectory[i]
        ex, ey, ez = actor_edge_coords(actor)   # world-frame edges this frame
        arrow_len  = 0.8

        # ── LEFT: orthographic world view ─────────────────────────────────────
        p_world = plot(;
            title            = "World View  |  Frame $i / $num_frames",
            proj_type        = :ortho,
            camera           = world_camera,
            legend           = :topright,
            background_color = :white,
            axis_kw...
        )

        plot!(p_world, actor_xs[1:i], actor_ys[1:i], actor_zs[1:i],
              linewidth=3, color=:blue, alpha=0.6, label="Actor Path")
        plot!(p_world, drone_xs[1:i], drone_ys[1:i], drone_zs[1:i],
              linewidth=3, color=:red, alpha=0.6, label="Drone Path")
        plot!(p_world, ex, ey, ez,
              color=:black, linewidth=2, label=:none)
        plot!(p_world,
              [actor.x, actor.x + arrow_len * cos(actor.heading)],
              [actor.y, actor.y + arrow_len * sin(actor.heading)],
              [actor.z, actor.z],
              color=:green, linewidth=3, arrow=true, label="Heading")
        draw_quadcopter!(p_world, drone, 0.3, 0.15)
        plot!(p_world,
              [drone[1], actor.x], [drone[2], actor.y], [drone[3], actor.z],
              linestyle=:dash, color=:gray, linewidth=1, alpha=0.4, label="LoS")
        scatter!(p_world, [actor_xs[1]], [actor_ys[1]], [actor_zs[1]],
                 markersize=6, color=:lightblue, marker=:square, label="Actor Start")
        scatter!(p_world, [drone_xs[1]], [drone_ys[1]], [drone_zs[1]],
                 markersize=6, color=:pink, marker=:square, label="Drone Start")

        # ── RIGHT: perspective FPV ────────────────────────────────────────────
        # Transform everything into drone-local frame:
        #   1. Subtract drone position  → drone at origin
        #   2. Rotate by -yaw           → drone always faces +X
        # Then camera=(0, fpv_tilt) gives true FPV — looking straight ahead.

        ex_l, ey_l, ez_l = edges_to_drone_frame(ex, ey, ez, drone)

        trail_xl, trail_yl, trail_zl = to_drone_frame(
            actor_xs[1:i], actor_ys[1:i], actor_zs[1:i], drone)

        hx_w = actor.x + arrow_len * cos(actor.heading)
        hy_w = actor.y + arrow_len * sin(actor.heading)
        arr_xl, arr_yl, arr_zl = to_drone_frame(
            [actor.x, hx_w], [actor.y, hy_w], [actor.z, actor.z], drone)

        fv   = 8.0   # view half-distance in metres
        dist = norm([actor.x - drone[1], actor.y - drone[2], actor.z - drone[3]])

        p_fpv = plot(;
            title            = "FPV  |  Frame $i / $num_frames",
            proj_type        = :persp,
            camera           = (0, fpv_tilt),
            background_color = :black,
            foreground_color = :white,
            legend           = false,
            xlabel = "X (m)", ylabel = "Y (m)", zlabel = "Z (m)",
            xlims  = (-fv, fv),
            ylims  = (-fv, fv),
            zlims  = (-fv, fv),
            aspect_ratio = :equal,
            grid   = true,
        )

        plot!(p_fpv, trail_xl, trail_yl, trail_zl,
              linewidth=3, color=:blue, alpha=0.8, label=:none)
        plot!(p_fpv, ex_l, ey_l, ez_l,
              color=:white, linewidth=2, label=:none)
        plot!(p_fpv, arr_xl, arr_yl, arr_zl,
              color=:green, linewidth=3, arrow=true, label=:none)

        # annotate!(p_fpv, -fv+0.3, -fv+0.3, -fv+0.3,
        #           text("● REC", :left, :red, 9))
        # annotate!(p_fpv, -fv+0.3,  fv-0.3, -fv+0.3,
        #           text("DST $(round(dist, digits=1))m", :left, :lime, 9))

        # ── Combine ───────────────────────────────────────────────────────────
        plot(p_world, p_fpv, layout=(1, 2), size=(1400, 700))
    end

    println("Saving animation → $anim_file …")
    gif(anim, anim_file, fps=fps)
    return anim
end

end # module