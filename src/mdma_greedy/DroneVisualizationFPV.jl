module DroneVisualizationFPV

using Plots
using LinearAlgebra
using ..MPC.ActorMesh: actor_world_vertices, actor_world_face_center, actor_world_normal

export draw_quadcopter!, animate_drone_and_actor, animate_multi_actor

# #This idea came from the camera, 
# function actor_edge_coords(actor)
#     wv = actor_world_vertices(actor.mesh, actor.x, actor.y, actor.z, actor.heading)
#     x  = [[wv[i1][1], wv[i2][1]] for (i1, i2) in actor.mesh.edges]
#     y  = [[wv[i1][2], wv[i2][2]] for (i1, i2) in actor.mesh.edges]
#     z  = [[wv[i1][3], wv[i2][3]] for (i1, i2) in actor.mesh.edges]
#     return x, y, z, wv
# end


#This function projects a single world-frame into the drone's perspective

function project_to_fpv(
    world_pt::Vector{Float64},
    drone::Vector{Float64};
    tilt_angle::Float64   = -0.35,
    focal_length::Float64 =  1.2
)

    #Here we are translating from the world frame to the drone frame
    dx = world_pt[1] - drone[1]
    dy = world_pt[2] - drone[2]
    dz = world_pt[3] - drone[3]

    yaw = drone[7]

    # As the fpv is 2D, the drone needs to - 
    bx =  dx * cos(yaw) + dy * sin(yaw) #go forward with actor
    by = -dx * sin(yaw) + dy * cos(yaw) #go further left, as it is rotating around yaw angle
    bz =  dz


    # we are adding a gimbal on top of the drone, so we need to calculate the gimbal distance from the drone body
    # tilt camera downward (rotation around body-Y axis)
    cx =  bx * cos(tilt_angle) + bz * sin(tilt_angle)   # forward depth of the camera
    cy =  by                                              # lateral
    cz = -bx * sin(tilt_angle) + bz * cos(tilt_angle)   # vertical in cam frame

    cx < 0.05 && return nothing, false    # behind or too close

    # Pinhole camera model
    u =  focal_length * cy / cx
    v =  focal_length * cz / cx

    return [u, v], true
end

function project_vertices_fpv(world_verts, drone; kwargs...)
    [let (pt, ok) = project_to_fpv(v, drone; kwargs...); ok ? pt : nothing end #Array comprehension
     for v in world_verts]
end

# 
#  FPV PANEL
# 

# draw_fpv_panel! — multi-actor version
# actors      : all ActorState objects at this frame
# primary_idx : index into actors that the drone is targeting
# trail_xs/ys/zs : past positions of the primary actor
function draw_fpv_panel!(p, actors::Vector, primary_idx::Int, drone,
                         trail_xs, trail_ys, trail_zs;
                         tilt_angle=-0.35, focal_length=1.2,
                         grid_range=-8:2:8)

    kw = (tilt_angle=tilt_angle, focal_length=focal_length)

    # Ground grid
    for gx in grid_range
        pa, ok_a = project_to_fpv([Float64(gx), Float64(first(grid_range)), 0.0], drone; kw...)
        pb, ok_b = project_to_fpv([Float64(gx), Float64(last(grid_range)),  0.0], drone; kw...)
        ok_a && ok_b && plot!(p, [pa[1], pb[1]], [pa[2], pb[2]],
                              color=:gray40, linewidth=0.8, alpha=0.5, label="")
    end
    for gy in grid_range
        pa, ok_a = project_to_fpv([Float64(first(grid_range)), Float64(gy), 0.0], drone; kw...)
        pb, ok_b = project_to_fpv([Float64(last(grid_range)),  Float64(gy), 0.0], drone; kw...)
        ok_a && ok_b && plot!(p, [pa[1], pb[1]], [pa[2], pb[2]],
                              color=:gray40, linewidth=0.8, alpha=0.5, label="")
    end

    # Axis lines
    ax_len = Float64(last(grid_range))
    for (tip, col) in [([ax_len, 0.0, 0.0], :red),
                        ([0.0, ax_len, 0.0], :green),
                        ([0.0, 0.0, ax_len], :dodgerblue)]
        pa, ok_a = project_to_fpv([0.0, 0.0, 0.0], drone; kw...)
        pb, ok_b = project_to_fpv(tip,              drone; kw...)
        ok_a && ok_b && plot!(p, [pa[1], pb[1]], [pa[2], pb[2]],
                              color=col, linewidth=1.5, alpha=0.7, label="")
    end

    # Primary actor trail
    trail_pts = []
    for (tx, ty, tz) in zip(trail_xs, trail_ys, trail_zs)
        pt, ok = project_to_fpv([tx, ty, tz], drone; kw...)
        push!(trail_pts, ok ? pt : nothing)
    end
    for k in 2:length(trail_pts)
        p1, p2 = trail_pts[k-1], trail_pts[k]
        (p1 === nothing || p2 === nothing) && continue
        plot!(p, [p1[1], p2[1]], [p1[2], p2[2]],
              color=:royalblue, linewidth=2.0, alpha=0.8, label="")
    end
    if !isempty(trail_pts) && trail_pts[1] !== nothing
        scatter!(p, [trail_pts[1][1]], [trail_pts[1][2]],
                 markersize=5, color=:lightblue, markerstrokewidth=0, label="")
    end

    # All actors: draw mesh in FPV. Targeting box only on primary.
    for (a_idx, actor) in enumerate(actors)
        world_verts = actor_world_vertices(
            actor.mesh, actor.x, actor.y, actor.z, actor.heading)
        proj = project_vertices_fpv(world_verts, drone; kw...)

        edge_col = (a_idx == primary_idx) ? :white : :gray70
        for (i1, i2) in actor.mesh.edges
            p1, p2 = proj[i1], proj[i2]
            (p1 === nothing || p2 === nothing) && continue
            plot!(p, [p1[1], p2[1]], [p1[2], p2[2]],
                  color=edge_col, linewidth=(a_idx == primary_idx ? 1.8 : 1.0), label="")
        end
        for face in actor.mesh.faces
            pts = [proj[idx] for idx in face.corner_indices]
            any(x -> x === nothing, pts) && continue
            us = [pt[1] for pt in pts]; push!(us, us[1])
            vs = [pt[2] for pt in pts]; push!(vs, vs[1])
            plot!(p, Shape(us, vs),
                  fillalpha=(a_idx == primary_idx ? 0.40 : 0.20),
                  fillcolor=face.color, linewidth=0)
        end

        # Targeting box — primary actor only (Axis-Aligned bounding box)
        if a_idx == primary_idx
            valid = filter(x -> x !== nothing, proj)
            if length(valid) >= 4
                us_a = [pt[1] for pt in valid]
                vs_a = [pt[2] for pt in valid]
                pad  = 0.025
                u0, u1 = minimum(us_a) - pad, maximum(us_a) + pad
                v0, v1 = minimum(vs_a) - pad, maximum(vs_a) + pad
                plot!(p, [u0, u1, u1, u0, u0], [v0, v0, v1, v1, v0],
                      color=:blue, linewidth=1.6, linestyle=:dash, alpha=0.9, label="")
                annotate!(p, u0, v1 + 0.035, text("TARGET", :left, :blue, 7))
            end
        end
    end

    # HUD crosshair
    ch, gap = 0.055, 0.015
    for (x1, x2, y1, y2) in [( gap,  ch,  0.0,  0.0),
                               (-ch, -gap,  0.0,  0.0),
                               ( 0.0,  0.0,  gap,  ch),
                               ( 0.0,  0.0, -ch, -gap)]
        plot!(p, [x1, x2], [y1, y2], color=:black, linewidth=2, alpha=0.9, label="")
    end
    scatter!(p, [0.0], [0.0], markersize=3, color=:black, markerstrokewidth=0, label="")

    # HUD readouts — distance to primary actor
    primary = actors[primary_idx]
    dist     = norm([primary.x - drone[1], primary.y - drone[2], primary.z - drone[3]])
    tilt_deg = round(Int, tilt_angle * 180 / π)
    xl       = Plots.xlims(p)[1]
    annotate!(p, xl + 0.02, -0.44, text("DST  $(lpad(round(Int,dist*10)/10, 4))m", :left, :lime, 8))
    annotate!(p, xl + 0.02, -0.50, text("ALT  $(lpad(round(Int,drone[3]*10)/10, 4))m", :left, :lime, 8))
    annotate!(p, xl + 0.02, -0.56, text("TILT $(tilt_deg)°", :left, :blue, 8))
    annotate!(p, xl + 0.02, -0.62, text("ACT  #$(primary_idx)/$(length(actors))", :left, :yellow, 8))

    # Corner bracket reticle
    bx_h, by_h, bl = 0.60, 0.44, 0.07
    for (sx, sy) in [(1,1),(-1,1),(1,-1),(-1,-1)]
        plot!(p, [sx*bx_h, sx*bx_h, sx*(bx_h-bl)],
                 [sy*(by_h-bl), sy*by_h, sy*by_h],
              color=:black, linewidth=1.5, alpha=0.7, label="")
    end
end

# Backward-compat single-actor wrapper (keeps first_person_view.jl working)
function draw_fpv_panel!(p, actor, drone, trail_xs, trail_ys, trail_zs;
                         tilt_angle=-0.35, focal_length=1.2, grid_range=-8:2:8)
    draw_fpv_panel!(p, [actor], 1, drone, trail_xs, trail_ys, trail_zs;
                    tilt_angle=tilt_angle, focal_length=focal_length, grid_range=grid_range)
end

# 
#  WORLD-VIEW HELPERS
# 

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

function draw_colored_actor!(p, actor, world_verts)
    # Faces: color and corner_indices come directly from mesh.faces
    for face in actor.mesh.faces
        face_corners = [world_verts[idx] for idx in face.corner_indices]
        xs_f = [c[1] for c in face_corners]; push!(xs_f, xs_f[1])
        ys_f = [c[2] for c in face_corners]; push!(ys_f, ys_f[1])
        zs_f = [c[3] for c in face_corners]; push!(zs_f, zs_f[1])
        plot!(p, xs_f, ys_f, zs_f,
              fillrange=0, fillalpha=0.7, fillcolor=face.color,
              linewidth=2, linecolor=:black, label="")
    end

    # Edges: drawn from mesh.edges (no hardcoded list here either)
    for (i1, i2) in actor.mesh.edges
        v1, v2 = world_verts[i1], world_verts[i2]
        plot!(p, [v1[1], v2[1]], [v1[2], v2[2]], [v1[3], v2[3]],
              color=:black, linewidth=1.5, label="")
    end

    # Heading arrow
    arrow_len = 0.8
    hx = actor.x + arrow_len * cos(actor.heading)
    hy = actor.y + arrow_len * sin(actor.heading)
    plot!(p, [actor.x, hx], [actor.y, hy], [actor.z, actor.z],
          color=:green, linewidth=4, arrow=true, label="Heading")
end

# 
#  MAIN ANIMATION  –  World view LEFT  |  FPV RIGHT
# 


function animate_drone_and_actor(
    actor_trajectory::Vector,
    drone_trajectory::Vector{Vector{Float64}};
    anim_file::String      = "drone_actor_tracking.gif",
    fps::Int               = 10,
    xlims::Tuple           = (-4.0, 4.0),
    ylims::Tuple           = (-4.0, 4.0),
    zlims::Tuple           = ( 0.0, 4.0),
    camera::Tuple          = (25, 45),
    fpv_tilt::Float64      = -0.35,
    fpv_fov::Float64       =  1.2,
    fpv_view_size::Float64 =  0.68
)
    actor_xs = [a.x for a in actor_trajectory]
    actor_ys = [a.y for a in actor_trajectory]
    actor_zs = [a.z for a in actor_trajectory]

    drone_xs = [d[1] for d in drone_trajectory]
    drone_ys = [d[2] for d in drone_trajectory]
    drone_zs = [d[3] for d in drone_trajectory]

    num_frames = min(length(actor_trajectory), length(drone_trajectory))
    vs = fpv_view_size

    anim = @animate for i in 1:num_frames
        actor = actor_trajectory[i]
        drone = drone_trajectory[i]

        # World-frame vertices for this frame (computed once, used by both panels)
        world_verts = actor_world_vertices(
            actor.mesh, actor.x, actor.y, actor.z, actor.heading)

        #LEFT: 3-D world view
        p_world = plot(
            xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
            title="World View  |  Frame $i / $num_frames",
            legend=:topright,
            camera=camera,
            xlims=xlims, ylims=ylims, zlims=zlims,
            background_color=:white,
            size=(700, 700)
        )

        plot!(p_world, actor_xs[1:i], actor_ys[1:i], actor_zs[1:i],
              linewidth=3, color=:blue, alpha=0.6, label="Actor Path")
        plot!(p_world, drone_xs[1:i], drone_ys[1:i], drone_zs[1:i],
              linewidth=3, color=:red,  alpha=0.6, label="Drone Path")

        draw_colored_actor!(p_world, actor, world_verts)
        draw_quadcopter!(p_world, drone, 0.3, 0.15)

        plot!(p_world,
              [drone[1], actor.x], [drone[2], actor.y], [drone[3], actor.z],
              linestyle=:dash, color=:gray, linewidth=1, alpha=0.4, label="LoS")

        scatter!(p_world, [actor_xs[1]], [actor_ys[1]], [actor_zs[1]],
                 markersize=6, color=:lightblue, marker=:square, label="Actor Start")
        scatter!(p_world, [drone_xs[1]], [drone_ys[1]], [drone_zs[1]],
                 markersize=6, color=:pink, marker=:square, label="Drone Start")

        # RIGHT: FPV panel 
        p_fpv = plot(
            title="FPV Camera",
            legend=false,
            xlims=(-vs, vs),
            ylims=(-vs * 0.72, vs * 0.72),
            aspect_ratio=:equal,
            background_color=:black,
            foreground_color_axis=:white,
            foreground_color_border=:black,
            grid=false, ticks=false, framestyle=:box,
            size=(700, 700)
        )

        draw_fpv_panel!(p_fpv, actor, drone,
                        actor_xs[1:i], actor_ys[1:i], actor_zs[1:i];
                        tilt_angle=fpv_tilt, focal_length=fpv_fov)

        annotate!(p_fpv, vs - 0.02, -vs*0.72 + 0.04,
                  text("FPV · DRONE CAM", :right, :black, 8))
        annotate!(p_fpv, vs - 0.02,  vs*0.72 - 0.04,
                  text("● REC", :right, :red, 8))

        # Combine
        plot(p_world, p_fpv, layout=(1, 2), size=(1400, 700))
    end

    println("Saving animation → $anim_file …")
    gif(anim, anim_file, fps=fps)
    return anim
end

function animate_multi_actor(
    all_actor_trajs::Vector,
    drone_trajectory::Vector{Vector{Float64}};
    primary_actor_idx::Int = 1,
    anim_file::String      = "multi_actor_tracking.gif",
    fps::Int               = 10,
    xlims::Tuple           = (-6.0, 6.0),
    ylims::Tuple           = (-6.0, 6.0),
    zlims::Tuple           = ( 0.0, 4.0),
    camera::Tuple          = (25, 45),
    fpv_tilt::Float64      = -0.35,
    fpv_fov::Float64       =  1.2,
    fpv_view_size::Float64 =  0.68
)
    num_actors = length(all_actor_trajs)
    num_frames = min(minimum(length.(all_actor_trajs)), length(drone_trajectory))
    vs = fpv_view_size

    # Pre-extract primary actor trail for FPV draw
    primary_traj = all_actor_trajs[primary_actor_idx]
    primary_xs = [a.x for a in primary_traj]
    primary_ys = [a.y for a in primary_traj]
    primary_zs = [a.z for a in primary_traj]

    drone_xs = [d[1] for d in drone_trajectory]
    drone_ys = [d[2] for d in drone_trajectory]
    drone_zs = [d[3] for d in drone_trajectory]

    anim = @animate for i in 1:num_frames
        drone = drone_trajectory[i]
        current_actors = [traj[i] for traj in all_actor_trajs]
        primary_actor = current_actors[primary_actor_idx]

        p_world = plot(
            xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
            title="World View  |  Frame $i / $num_frames",
            legend=:topright, camera=camera,
            xlims=xlims, ylims=ylims, zlims=zlims,
            background_color=:white, size=(700, 700)
        )

        plot!(p_world, drone_xs[1:i], drone_ys[1:i], drone_zs[1:i],
              linewidth=3, color=:red,  alpha=0.6, label="Drone Path")
        draw_quadcopter!(p_world, drone, 0.3, 0.15)

        for (a_idx, actor) in enumerate(current_actors)
            traj = all_actor_trajs[a_idx]
            axs = [a.x for a in traj]
            ays = [a.y for a in traj]
            azs = [a.z for a in traj]
            
            # Draw path
            col = (a_idx == primary_actor_idx) ? :blue : :orange
            plot!(p_world, axs[1:i], ays[1:i], azs[1:i],
                  linewidth=(a_idx == primary_actor_idx ? 3 : 2), 
                  color=col, alpha=0.6, label=(a_idx == primary_actor_idx ? "Primary Path" : "Actor Path"))
            
            # Start position
            scatter!(p_world, [axs[1]], [ays[1]], [azs[1]],
                     markersize=5, color=col, marker=:square, label="")
            
            # Draw actor mesh
            world_verts = actor_world_vertices(actor.mesh, actor.x, actor.y, actor.z, actor.heading)
            draw_colored_actor!(p_world, actor, world_verts)
        end

        # Line of sight
        plot!(p_world,
              [drone[1], primary_actor.x], [drone[2], primary_actor.y], [drone[3], primary_actor.z],
              linestyle=:dash, color=:gray, linewidth=1, alpha=0.4, label="LoS")

        scatter!(p_world, [drone_xs[1]], [drone_ys[1]], [drone_zs[1]],
                 markersize=6, color=:pink, marker=:square, label="Drone Start")

        # RIGHT: FPV panel 
        p_fpv = plot(
            title="FPV Camera", legend=false,
            xlims=(-vs, vs), ylims=(-vs * 0.72, vs * 0.72),
            aspect_ratio=:equal, background_color=:black,
            foreground_color_axis=:white, foreground_color_border=:black,
            grid=false, ticks=false, framestyle=:box, size=(700, 700)
        )

        draw_fpv_panel!(p_fpv, current_actors, primary_actor_idx, drone,
                        primary_xs[1:i], primary_ys[1:i], primary_zs[1:i];
                        tilt_angle=fpv_tilt, focal_length=fpv_fov)

        annotate!(p_fpv, vs - 0.02, -vs*0.72 + 0.04, text("FPV · DRONE CAM", :right, :black, 8))
        annotate!(p_fpv, vs - 0.02,  vs*0.72 - 0.04, text("● REC", :right, :red, 8))

        plot(p_world, p_fpv, layout=(1, 2), size=(1400, 700))
    end

    println("Saving animation → $anim_file …")
    gif(anim, anim_file, fps=fps)
    return anim
end

end # module