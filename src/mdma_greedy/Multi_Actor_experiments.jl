# Multi_Actor_experiments.jl
# Four multi-actor / single-drone framing scenarios, reusing the visualization
# design from Single_Actor_experiments.jl (World View 3D, FPV HUD camera,
# Smooth PPA heatmap). Both actors sit at the SAME fixed ground positions in
# every experiment — only the drone's position/behavior changes per scenario.
#
#   1. Close drone shot of both actors' faces
#   2. Drone backs away just far enough to keep both actors' full bodies in frame
#   3. Actor 1 always kept in frame, even when actor 2 is not
#   4. Both actors prefer their own front face; the drone focuses on whichever
#      actor's front face currently yields more PPA

ENV["GKSwstype"] = "100"

if !@isdefined(ActorMeshTriangulated)
    include(joinpath(@__DIR__, "ActorMeshTriangulated.jl"))
end

using .ActorMeshTriangulated
using Plots
using LinearAlgebra

const OBJ_PATH = joinpath(@__DIR__, "simple_human_rotated_color.obj")
const PART_DECAY = Dict(:Head_face => 2.0, :Body_face => 1.0, :Feet_face => 0.5, :Top_face => 1.5)

const TILT = -0.35
const FOCAL_LENGTH = 1.2
const FPV_VIEW_SIZE = 0.68

# Fixed actor ground positions — identical across all 4 experiments.
const ACTOR1_XY = (-2.0, 2.0)
const ACTOR2_XY = (-2.0, -2.0)

struct ActorPlacement
    x::Float64
    y::Float64
    heading::Float64
end

# ── Camera projection (parameterized by drone pose, since the drone moves
# between experiments here — unlike Single_Actor_experiments.jl's fixed drone) ─
function project_point(v_world::Vector{Float64}, drone_pos::Vector{Float64}, drone_yaw::Float64)
    dx = v_world[1] - drone_pos[1]; dy = v_world[2] - drone_pos[2]; dz = v_world[3] - drone_pos[3]
    bx = dx*cos(drone_yaw) + dy*sin(drone_yaw)
    by = -dx*sin(drone_yaw) + dy*cos(drone_yaw)
    cx_raw = bx*cos(TILT) + dz*sin(TILT)
    cz_raw = -bx*sin(TILT) + dz*cos(TILT)
    cx_denom = max(cx_raw, 0.1)
    u = FOCAL_LENGTH * by / cx_denom
    v_ = FOCAL_LENGTH * cz_raw / cx_denom
    return u, v_, cx_raw
end

function project_or_nothing(v_world::Vector{Float64}, drone_pos::Vector{Float64}, drone_yaw::Float64)
    u, v_, cx = project_point(v_world, drone_pos, drone_yaw)
    cx < 0.05 && return nothing
    return (u, v_)
end

in_frame(u, v_, cx) = cx > 0.05 && abs(u) <= 1.0 && abs(v_) <= 1.0

function actor_center_uv(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64,
                          drone_pos::Vector{Float64}, drone_yaw::Float64)
    return project_point([x, y, z + mesh.height/2], drone_pos, drone_yaw)
end

function whole_mesh_in_frame(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64,
                              drone_pos::Vector{Float64}, drone_yaw::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    for v in world_verts
        u, v_, cx = project_point(v, drone_pos, drone_yaw)
        in_frame(u, v_, cx) || return false
    end
    return true
end

function head_fully_in_frame(mesh::TriMeshStruct, x::Float64, y::Float64, heading::Float64,
                              drone_pos::Vector{Float64}, drone_yaw::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, 0.0, heading)
    for face in mesh.faces
        face.name == :Head_face || continue
        for idx in face.corner_indices
            u, v_, cx = project_point(world_verts[idx], drone_pos, drone_yaw)
            in_frame(u, v_, cx) || return false
        end
    end
    return true
end

part_color(name::Symbol) = name == :Head_face ? :orange :
                            name == :Body_face ? :royalblue :
                            name == :Feet_face ? :seagreen : :crimson

# ── World View (3D) ───────────────────────────────────────────────────────────
function draw_quadcopter3d!(p, drone_pos::Vector{Float64}, drone_yaw::Float64; arm_length=0.3, prop_radius=0.15)
    x, y, z = drone_pos
    scatter!(p, [x], [y], [z], markersize=8, color=:darkred,
             markerstrokewidth=2, markerstrokecolor=:black, label="")

    for (i, arm_offset) in enumerate([π/4, 3π/4, 5π/4, 7π/4])
        angle = drone_yaw + arm_offset
        arm_x = x + arm_length * cos(angle)
        arm_y = y + arm_length * sin(angle)
        plot!(p, [x, arm_x], [y, arm_y], [z, z], color=:black, linewidth=3, label="")
        scatter!(p, [arm_x], [arm_y], [z], markersize=4, color=:gray,
                 markerstrokewidth=1, markerstrokecolor=:black, label="")

        prop_θ = range(0, 2π; length=20)
        prop_x = arm_x .+ prop_radius .* cos.(prop_θ) .* cos(angle) .- prop_radius .* sin.(prop_θ) .* sin(angle)
        prop_y = arm_y .+ prop_radius .* cos.(prop_θ) .* sin(angle) .+ prop_radius .* sin.(prop_θ) .* cos(angle)
        plot!(p, prop_x, prop_y, fill(z + 0.05, length(prop_θ)),
              color=(i % 2 == 0 ? :red : :blue), linewidth=2, alpha=0.6, label="")
    end

    cam_x = x + 0.15 * cos(drone_yaw)
    cam_y = y + 0.15 * sin(drone_yaw)
    plot!(p, [x, cam_x], [y, cam_y], [z, z - 0.1], color=:lime, linewidth=3, label="")
    scatter!(p, [cam_x], [cam_y], [z - 0.1], markersize=3, color=:lime, marker=:square, label="")
end

function draw_actor_3d!(p, mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64; highlight::Bool=false)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    for face in mesh.faces
        corners = [world_verts[idx] for idx in face.corner_indices]
        xs_f = [c[1] for c in corners]; push!(xs_f, xs_f[1])
        ys_f = [c[2] for c in corners]; push!(ys_f, ys_f[1])
        zs_f = [c[3] for c in corners]; push!(zs_f, zs_f[1])
        # Plots.jl/GR does not support fillrange-based fill on 3D line series
        # (silently ignored), so color the triangle via its outline instead.
        plot!(p, xs_f, ys_f, zs_f, color=part_color(face.name), linewidth=(highlight ? 3.5 : 2.0), label="")
    end
    arrow_len = 0.8
    hx = x + arrow_len * cos(heading)
    hy = y + arrow_len * sin(heading)
    plot!(p, [x, hx], [y, hy], [z, z], color=(highlight ? :gold : :green), linewidth=4, arrow=true, label="")
end

# The 4 corners of the camera's rectangular FOV frustum at a fixed depth along
# its optical axis (same construction as Single_Actor_experiments.jl, now
# parameterized by drone pose).
function fov_frustum_corners(drone_pos::Vector{Float64}, drone_yaw::Float64, depth::Float64)
    f = FOCAL_LENGTH
    corners = Vector{Float64}[]
    for (su, sv) in [(-1,-1), (1,-1), (1,1), (-1,1)]
        s = sv * depth / f
        bx = depth*cos(TILT) - s*sin(TILT)
        by = su * depth / f
        dz = depth*sin(TILT) + s*cos(TILT)
        dx = bx*cos(drone_yaw) - by*sin(drone_yaw)
        dy = bx*sin(drone_yaw) + by*cos(drone_yaw)
        push!(corners, [drone_pos[1]+dx, drone_pos[2]+dy, drone_pos[3]+dz])
    end
    return corners
end

function draw_fov_frustum!(p, drone_pos::Vector{Float64}, drone_yaw::Float64; depth::Float64 = 2.0)
    corners = fov_frustum_corners(drone_pos, drone_yaw, depth)
    for c in corners
        plot!(p, [drone_pos[1], c[1]], [drone_pos[2], c[2]], [drone_pos[3], c[3]],
              color=:gray, linestyle=:dash, linewidth=1, alpha=0.6, label="")
    end
    xs_f = [c[1] for c in corners]; push!(xs_f, xs_f[1])
    ys_f = [c[2] for c in corners]; push!(ys_f, ys_f[1])
    zs_f = [c[3] for c in corners]; push!(zs_f, zs_f[1])
    plot!(p, xs_f, ys_f, zs_f, color=:gray, linestyle=:dash, linewidth=1, alpha=0.6, label="")
end

function build_world_view(mesh::TriMeshStruct, actors::Vector{ActorPlacement},
                           drone_pos::Vector{Float64}, drone_yaw::Float64; focus_idx::Int = 0)
    p_world = plot(
        xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="World View",
        legend=false, camera=(30, 45), aspect_ratio=:equal,
        xlims=(-8.0, 8.0), ylims=(-8.0, 8.0), zlims=(0.0, 5.0),
        background_color=:white
    )
    for (i, a) in enumerate(actors)
        draw_actor_3d!(p_world, mesh, a.x, a.y, 0.0, a.heading; highlight=(i == focus_idx))
        mesh_center_z = mesh.height/2
        plot!(p_world, [drone_pos[1], a.x], [drone_pos[2], a.y], [drone_pos[3], mesh_center_z],
              linestyle=:dash, color=:gray, linewidth=1, alpha=0.6, label="")
    end
    draw_quadcopter3d!(p_world, drone_pos, drone_yaw)
    draw_fov_frustum!(p_world, drone_pos, drone_yaw)
    return p_world
end

# ── FPV Camera (black HUD-style view) ─────────────────────────────────────────
function draw_ground_grid_and_axes!(p, drone_pos::Vector{Float64}, drone_yaw::Float64; grid_range=-8:2:8)
    for gx in grid_range
        pa = project_or_nothing([Float64(gx), Float64(first(grid_range)), 0.0], drone_pos, drone_yaw)
        pb = project_or_nothing([Float64(gx), Float64(last(grid_range)), 0.0], drone_pos, drone_yaw)
        (pa === nothing || pb === nothing) || plot!(p, [pa[1],pb[1]], [pa[2],pb[2]], color=:gray40, linewidth=0.8, alpha=0.5, label="")
    end
    for gy in grid_range
        pa = project_or_nothing([Float64(first(grid_range)), Float64(gy), 0.0], drone_pos, drone_yaw)
        pb = project_or_nothing([Float64(last(grid_range)), Float64(gy), 0.0], drone_pos, drone_yaw)
        (pa === nothing || pb === nothing) || plot!(p, [pa[1],pb[1]], [pa[2],pb[2]], color=:gray40, linewidth=0.8, alpha=0.5, label="")
    end
    ax_len = Float64(last(grid_range))
    origin = project_or_nothing([0.0, 0.0, 0.0], drone_pos, drone_yaw)
    for (tip, col) in [([ax_len,0.0,0.0],:red), ([0.0,ax_len,0.0],:green), ([0.0,0.0,ax_len],:dodgerblue)]
        pb = project_or_nothing(tip, drone_pos, drone_yaw)
        (origin === nothing || pb === nothing) || plot!(p, [origin[1],pb[1]], [origin[2],pb[2]], color=col, linewidth=1.5, alpha=0.7, label="")
    end
end

function draw_fpv_actor!(p, mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64,
                          drone_pos::Vector{Float64}, drone_yaw::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    all_pts = Tuple{Float64,Float64}[]
    for face in mesh.faces
        proj = [project_or_nothing(world_verts[idx], drone_pos, drone_yaw) for idx in face.corner_indices]
        any(pt -> pt === nothing, proj) && continue
        us = [pt[1] for pt in proj]; vs = [pt[2] for pt in proj]
        plot!(p, Shape(us, vs), fillalpha=0.65, fillcolor=part_color(face.name),
              linecolor=:black, linewidth=0.5, label="")
        append!(all_pts, zip(us, vs))
    end
    return all_pts
end

function draw_targeting_box!(p, pts::Vector{Tuple{Float64,Float64}}; color=:cyan, label_text="TARGET")
    isempty(pts) && return
    us = first.(pts); vs = last.(pts)
    pad = 0.025
    u0, u1 = minimum(us)-pad, maximum(us)+pad
    v0, v1 = minimum(vs)-pad, maximum(vs)+pad
    plot!(p, [u0,u1,u1,u0,u0], [v0,v0,v1,v1,v0], color=color, linewidth=1.6, linestyle=:dash, alpha=0.9, label="")
    annotate!(p, u0, v1+0.035, text(label_text, :left, color, 7))
end

function draw_hud_decorations!(p, ref_dist::Float64, drone_pos::Vector{Float64}, vs::Float64)
    ch, gap = 0.055, 0.015
    for (x1,x2,y1,y2) in [(gap,ch,0.0,0.0), (-ch,-gap,0.0,0.0), (0.0,0.0,gap,ch), (0.0,0.0,-ch,-gap)]
        plot!(p, [x1,x2], [y1,y2], color=:white, linewidth=2, alpha=0.9, label="")
    end
    scatter!(p, [0.0], [0.0], markersize=3, color=:white, markerstrokewidth=0, label="")

    bx_h, by_h, bl = 0.60, vs*0.72*0.85, 0.07
    for (sx, sy) in [(1,1),(-1,1),(1,-1),(-1,-1)]
        plot!(p, [sx*bx_h, sx*bx_h, sx*(bx_h-bl)], [sy*(by_h-bl), sy*by_h, sy*by_h],
              color=:white, linewidth=1.5, alpha=0.7, label="")
    end

    tilt_deg = round(Int, TILT * 180 / π)
    xl = -vs
    annotate!(p, xl+0.02, -vs*0.72+0.06, text("DST  $(round(ref_dist,digits=1))m", :left, :lime, 7))
    annotate!(p, xl+0.02, -vs*0.72+0.12, text("ALT  $(round(drone_pos[3],digits=1))m", :left, :lime, 7))
    annotate!(p, xl+0.02, -vs*0.72+0.18, text("TILT $(tilt_deg)°", :left, :dodgerblue, 7))
    annotate!(p, vs-0.02, -vs*0.72+0.05, text("FPV · DRONE CAM", :right, :white, 8))
    annotate!(p, vs-0.02, vs*0.72-0.04, text("● REC", :right, :red, 8))
end

function build_fpv_view(mesh::TriMeshStruct, actors::Vector{ActorPlacement},
                         drone_pos::Vector{Float64}, drone_yaw::Float64; focus_idx::Int = 0)
    vs = FPV_VIEW_SIZE
    p_fpv = plot(title="FPV Camera", legend=false,
        xlims=(-vs, vs), ylims=(-vs*0.72, vs*0.72), aspect_ratio=:equal,
        background_color=:black, foreground_color_axis=:white, foreground_color_border=:black,
        grid=false, ticks=false, framestyle=:box)
    draw_ground_grid_and_axes!(p_fpv, drone_pos, drone_yaw)

    for (i, a) in enumerate(actors)
        pts = draw_fpv_actor!(p_fpv, mesh, a.x, a.y, 0.0, a.heading, drone_pos, drone_yaw)
        label_text = focus_idx == i ? "FOCUS" : "ACTOR $i"
        box_color = focus_idx == i ? :gold : :cyan
        draw_targeting_box!(p_fpv, pts; color=box_color, label_text=label_text)
    end

    ref_dist = minimum(norm([a.x - drone_pos[1], a.y - drone_pos[2], 0.0 - drone_pos[3]]) for a in actors)
    draw_hud_decorations!(p_fpv, ref_dist, drone_pos, vs)
    return p_fpv
end

# ── Smooth PPA (matches the current "require ALL vertices in FOV" rule in
# StaticPPA_OrigVsSmooth_FOV.jl) ──────────────────────────────────────────────
function shoelace_signed(pts::Vector)
    n = length(pts); A = 0.0
    for i in 1:n
        j = mod1(i+1, n)
        A += pts[i][1]*pts[j][2] - pts[j][1]*pts[i][2]
    end
    return A / 2.0
end

function smooth_ppa_value(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64,
                           drone_pos::Vector{Float64}, drone_yaw::Float64)
    actor_pos = [x, y]
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    ps = 0.0
    for face in mesh.faces
        c, s = cos(heading), sin(heading)
        n_world = [face.normal[1]*c - face.normal[2]*s, face.normal[1]*s + face.normal[2]*c, face.normal[3]]
        weight = ActorMeshTriangulated.face_dynamic_weight(face, heading, actor_pos, drone_pos[1:2])

        uv = Vector{Vector{Float64}}(); w = 1.0
        face_pos = sum(world_verts[idx] for idx in face.corner_indices) / length(face.corner_indices)
        dist = face_pos .- drone_pos
        n_dot = -dot(n_world, dist)
        is_occ = n_dot <= 0.0
        vis_n = 0
        for idx in face.corner_indices
            v = world_verts[idx]
            dx = v[1]-drone_pos[1]; dy = v[2]-drone_pos[2]; dz = v[3]-drone_pos[3]
            bx = dx*cos(drone_yaw)+dy*sin(drone_yaw); by = -dx*sin(drone_yaw)+dy*cos(drone_yaw)
            cx_raw = bx*cos(TILT)+dz*sin(TILT); cz_raw = -bx*sin(TILT)+dz*cos(TILT)
            cx_soft = (cx_raw+sqrt(cx_raw^2+1e-4))/2.0
            w *= cx_soft/(cx_soft+0.1)
            cx_denom = max(cx_raw, 0.1)
            u = FOCAL_LENGTH*by/cx_denom; v_ = FOCAL_LENGTH*cz_raw/cx_denom
            push!(uv, [u, v_])
            if !is_occ && cx_raw > 0.05 && abs(u) <= 1.0 && abs(v_) <= 1.0
                vis_n += 1
            end
        end
        w *= vis_n >= length(face.corner_indices) ? 1.0 : 0.0
        vis_soft = (n_dot + sqrt(n_dot^2 + 1e-4)) / 2.0
        A_mag = abs(shoelace_signed(uv))
        ps += weight * w * vis_soft * A_mag
    end
    return ps
end

# Holds BOTH actors FIXED at their actual position + heading for this
# experiment, and sweeps the DRONE's ground position instead — at each
# candidate drone position the drone yaws toward the actors' midpoint, and the
# combined (summed) PPA of both actors is evaluated there. This shows how PPA
# responds to camera placement for this concrete actor configuration, with a
# smooth gradient around the actors rather than a single painted point.
function compute_ppa_over_drone_positions(mesh::TriMeshStruct, actors::Vector{ActorPlacement},
                                           drone_altitude::Float64;
                                           grid_size::Int = 101)
    xs = range(-8.0, 8.0, length=grid_size)
    ys = range(-8.0, 8.0, length=grid_size)
    mid_x = sum(a.x for a in actors) / length(actors)
    mid_y = sum(a.y for a in actors) / length(actors)
    per_actor_grids = [
        [smooth_ppa_value(mesh, a.x, a.y, 0.0, a.heading, [dx, dy, drone_altitude], atan(mid_y-dy, mid_x-dx))
         for dy in ys, dx in xs]
        for a in actors
    ]
    grid = sum(per_actor_grids)
    return xs, ys, grid, per_actor_grids
end

# ── Experiment 1: closest drone position where BOTH actors' heads are fully
# in frame — a "close shot of both faces." ────────────────────────────────────
function search_close_shot_both_faces(mesh::TriMeshStruct;
                                       drone_z::Float64 = 1.4,
                                       xs = range(-6.0, 3.0, length=91), ys = range(-6.0, 6.0, length=121))
    mid_x = (ACTOR1_XY[1] + ACTOR2_XY[1]) / 2
    mid_y = (ACTOR1_XY[2] + ACTOR2_XY[2]) / 2
    best_dist = Inf
    best_xy = (0.0, 0.0)
    for dx in xs, dy in ys
        drone_pos = [dx, dy, drone_z]
        drone_yaw = atan(mid_y-dy, mid_x-dx)
        h1 = atan(dy-ACTOR1_XY[2], dx-ACTOR1_XY[1])
        h2 = atan(dy-ACTOR2_XY[2], dx-ACTOR2_XY[1])
        head_fully_in_frame(mesh, ACTOR1_XY[1], ACTOR1_XY[2], h1, drone_pos, drone_yaw) || continue
        head_fully_in_frame(mesh, ACTOR2_XY[1], ACTOR2_XY[2], h2, drone_pos, drone_yaw) || continue
        d = norm([dx, dy] .- [mid_x, mid_y])
        if d < best_dist
            best_dist = d
            best_xy = (dx, dy)
        end
    end
    return best_xy, drone_z
end

# ── Experiment 2: closest drone position where BOTH actors' WHOLE bodies are
# fully in frame — the drone backs away just far enough to fit both. ─────────
function search_full_body_both(mesh::TriMeshStruct;
                                drone_z::Float64 = 3.0,
                                xs = range(-7.0, 4.0, length=111), ys = range(-8.0, 8.0, length=161))
    mid_x = (ACTOR1_XY[1] + ACTOR2_XY[1]) / 2
    mid_y = (ACTOR1_XY[2] + ACTOR2_XY[2]) / 2
    best_dist = Inf
    best_xy = (0.0, 0.0)
    for dx in xs, dy in ys
        drone_pos = [dx, dy, drone_z]
        drone_yaw = atan(mid_y-dy, mid_x-dx)
        h1 = atan(dy-ACTOR1_XY[2], dx-ACTOR1_XY[1])
        h2 = atan(dy-ACTOR2_XY[2], dx-ACTOR2_XY[1])
        whole_mesh_in_frame(mesh, ACTOR1_XY[1], ACTOR1_XY[2], 0.0, h1, drone_pos, drone_yaw) || continue
        whole_mesh_in_frame(mesh, ACTOR2_XY[1], ACTOR2_XY[2], 0.0, h2, drone_pos, drone_yaw) || continue
        d = norm([dx, dy] .- [mid_x, mid_y])
        if d < best_dist
            best_dist = d
            best_xy = (dx, dy)
        end
    end
    return best_xy, drone_z
end

# ── Experiment 3: closest drone position where actor 1's WHOLE body is fully
# in frame, ignoring actor 2 entirely — a tight shot on actor 1 alone. ────────
function search_actor1_priority(mesh::TriMeshStruct;
                                 drone_z::Float64 = 2.0,
                                 xs = range(-6.0, 3.0, length=91), ys = range(-6.0, 6.0, length=121))
    best_dist = Inf
    best_xy = (0.0, 0.0)
    for dx in xs, dy in ys
        drone_pos = [dx, dy, drone_z]
        h1 = atan(dy-ACTOR1_XY[2], dx-ACTOR1_XY[1])
        drone_yaw = atan(ACTOR1_XY[2]-dy, ACTOR1_XY[1]-dx)
        whole_mesh_in_frame(mesh, ACTOR1_XY[1], ACTOR1_XY[2], 0.0, h1, drone_pos, drone_yaw) || continue
        d = norm([dx, dy] .- collect(ACTOR1_XY))
        if d < best_dist
            best_dist = d
            best_xy = (dx, dy)
        end
    end
    return best_xy, drone_z
end

# ── Experiment 4: both actors keep their own FIXED "front face" heading
# (independent of the drone), and the drone searches for the position that
# maximizes whichever actor's front face currently yields more PPA. ──────────
function search_focus_on_better_actor(mesh::TriMeshStruct, heading1::Float64, heading2::Float64;
                                       drone_z::Float64 = 3.0,
                                       xs = range(-2.0, 6.0, length=81), ys = range(-6.0, 6.0, length=121))
    best_val = -Inf
    best_xy = (0.0, 0.0)
    best_focus = 1
    mid_x = (ACTOR1_XY[1] + ACTOR2_XY[1]) / 2
    mid_y = (ACTOR1_XY[2] + ACTOR2_XY[2]) / 2
    for dx in xs, dy in ys
        drone_pos = [dx, dy, drone_z]
        drone_yaw = atan(mid_y-dy, mid_x-dx)
        ppa1 = smooth_ppa_value(mesh, ACTOR1_XY[1], ACTOR1_XY[2], 0.0, heading1, drone_pos, drone_yaw)
        ppa2 = smooth_ppa_value(mesh, ACTOR2_XY[1], ACTOR2_XY[2], 0.0, heading2, drone_pos, drone_yaw)
        val, focus = ppa1 >= ppa2 ? (ppa1, 1) : (ppa2, 2)
        if val > best_val
            best_val = val
            best_xy = (dx, dy)
            best_focus = focus
        end
    end
    return best_xy, drone_z, best_focus
end

# ── One experiment: World View + FPV Camera + Smooth PPA heatmap ────────────
function render_multi_actor_figure(mesh::TriMeshStruct, title::String, out_file::String,
                                    actors::Vector{ActorPlacement}, drone_pos::Vector{Float64}, drone_yaw::Float64,
                                    xs, ys, H, shared_max::Float64; focus_idx::Int = 0)
    p_world = build_world_view(mesh, actors, drone_pos, drone_yaw; focus_idx=focus_idx)
    p_fpv = build_fpv_view(mesh, actors, drone_pos, drone_yaw; focus_idx=focus_idx)

    ppa_values = [smooth_ppa_value(mesh, a.x, a.y, 0.0, a.heading, drone_pos, drone_yaw) for a in actors]

    p_heat = heatmap(xs, ys, H, c=:jet, clims=(0.0, shared_max), title="Smooth PPA vs Drone Position",
        xlabel="Drone X (m)", ylabel="Drone Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box)
    scatter!(p_heat, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="")
    quiver!(p_heat, [drone_pos[1]], [drone_pos[2]], quiver=([0.9*cos(drone_yaw)],[0.9*sin(drone_yaw)]), color=:white, lw=2.5)
    for (i, a) in enumerate(actors)
        mc = focus_idx == i ? :gold : :white
        scatter!(p_heat, [a.x], [a.y], shape=:circle, ms=7, mc=mc, msc=:black, label="")
        quiver!(p_heat, [a.x], [a.y], quiver=([0.8*cos(a.heading)],[0.8*sin(a.heading)]), color=mc, lw=2.0)
    end

    p_combined = plot(p_world, p_fpv, p_heat, layout=(1,3), size=(1800,600), plot_title=title)

    mkpath("src/mdma_greedy/drone_experiments")
    savefig(p_combined, out_file)
    println("✓ Saved → $out_file  (PPA1=$(round(ppa_values[1], digits=6)) PPA2=$(round(ppa_values[2], digits=6)))")
    return p_combined
end

function run_all_multi_actor_experiments()
    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)
    dir = "src/mdma_greedy/drone_experiments"

    # ── Experiment 1: Close shot of both faces ──────────────────────────────
    (dx1, dy1), dz1 = search_close_shot_both_faces(mesh)
    drone_pos1 = [dx1, dy1, dz1]
    drone_yaw1 = atan(((ACTOR1_XY[2]+ACTOR2_XY[2])/2)-dy1, ((ACTOR1_XY[1]+ACTOR2_XY[1])/2)-dx1)
    h1a = atan(dy1-ACTOR1_XY[2], dx1-ACTOR1_XY[1])
    h1b = atan(dy1-ACTOR2_XY[2], dx1-ACTOR2_XY[1])
    actors1 = [ActorPlacement(ACTOR1_XY[1], ACTOR1_XY[2], h1a), ActorPlacement(ACTOR2_XY[1], ACTOR2_XY[2], h1b)]

    # ── Experiment 2: Full body of both, drone flying away until it fits ────
    (dx2, dy2), dz2 = search_full_body_both(mesh)
    drone_pos2 = [dx2, dy2, dz2]
    drone_yaw2 = atan(((ACTOR1_XY[2]+ACTOR2_XY[2])/2)-dy2, ((ACTOR1_XY[1]+ACTOR2_XY[1])/2)-dx2)
    h2a = atan(dy2-ACTOR1_XY[2], dx2-ACTOR1_XY[1])
    h2b = atan(dy2-ACTOR2_XY[2], dx2-ACTOR2_XY[1])
    actors2 = [ActorPlacement(ACTOR1_XY[1], ACTOR1_XY[2], h2a), ActorPlacement(ACTOR2_XY[1], ACTOR2_XY[2], h2b)]

    # ── Experiment 3: Actor 1 always in frame, actor 2 not necessarily ──────
    (dx3, dy3), dz3 = search_actor1_priority(mesh)
    drone_pos3 = [dx3, dy3, dz3]
    drone_yaw3 = atan(ACTOR1_XY[2]-dy3, ACTOR1_XY[1]-dx3)
    h3a = atan(dy3-ACTOR1_XY[2], dx3-ACTOR1_XY[1])
    h3b = atan(dy3-ACTOR2_XY[2], dx3-ACTOR2_XY[1])
    actors3 = [ActorPlacement(ACTOR1_XY[1], ACTOR1_XY[2], h3a), ActorPlacement(ACTOR2_XY[1], ACTOR2_XY[2], h3b)]
    actor2_excluded = !whole_mesh_in_frame(mesh, ACTOR2_XY[1], ACTOR2_XY[2], 0.0, h3b, drone_pos3, drone_yaw3)
    #println("Experiment 3: actor 2 excluded from frame? $actor2_excluded")

    # ── Experiment 4: fixed front-face headings — actor 1 faces the general
    # drone side (+X), actor 2 faces away (-X) — deliberately asymmetric so the
    # "focus on whichever front face has more PPA" logic has a real winner. ──
    heading4a = 0.0        # actor 1's fixed front face: +X
    heading4b = Float64(pi) # actor 2's fixed front face: -X (away)
    (dx4, dy4), dz4, focus4 = search_focus_on_better_actor(mesh, heading4a, heading4b)
    drone_pos4 = [dx4, dy4, dz4]
    focus_xy = focus4 == 1 ? ACTOR1_XY : ACTOR2_XY
    drone_yaw4 = atan(focus_xy[2]-dy4, focus_xy[1]-dx4)
    actors4 = [ActorPlacement(ACTOR1_XY[1], ACTOR1_XY[2], heading4a), ActorPlacement(ACTOR2_XY[1], ACTOR2_XY[2], heading4b)]
    #println("Experiment 4: focusing on actor $focus4 (higher front-face PPA)")

    scenarios = [
        ("Close Shot — Both Faces",         "$dir/MultiActor_CloseBothFaces.png",  actors1, drone_pos1, drone_yaw1, 0),
        ("Full Body — Both Actors in Frame","$dir/MultiActor_FullBodyBoth.png",    actors2, drone_pos2, drone_yaw2, 0),
        ("Actor 1 Priority",                "$dir/MultiActor_Actor1Priority.png", actors3, drone_pos3, drone_yaw3, 1),
        ("Focus on Better-Oriented Actor",  "$dir/MultiActor_FocusBetterActor.png", actors4, drone_pos4, drone_yaw4, focus4),
    ]

    # First pass: compute every heatmap grid (actors fixed, drone position
    # swept) so we can share one color scale.
    grids = [compute_ppa_over_drone_positions(mesh, actors, drone_pos[3])
             for (_, _, actors, drone_pos, drone_yaw, _) in scenarios]
    #shared_max = max(maximum(maximum(g[3]) for g in grids), 1e-4)
    #println("Shared heatmap scale: 0.0 – $(round(shared_max, digits=4))")
    shared_max = 0.5

    for ((title, out_file, actors, drone_pos, drone_yaw, focus_idx), (xs, ys, H, per_actor_grids)) in zip(scenarios, grids)
        println("$title — highest achievable PPA over all swept drone positions:")
        for (i, g) in enumerate(per_actor_grids)
            println("  Actor $i: $(round(maximum(g), digits=6))")
        end
        render_multi_actor_figure(mesh, title, out_file, actors, drone_pos, drone_yaw, xs, ys, H, shared_max; focus_idx=focus_idx)
    end
end

run_all_multi_actor_experiments()
