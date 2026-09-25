# Single_Actor_experiments.jl
# Four single-actor / single-drone framing scenarios, each rendered as its own
# 3-panel figure: World View (3D, quadcopter + colored actor mesh), FPV Camera
# (black HUD-style camera view, matching the design of
# drone_experiments/drone_follows_actor_fpv.gif but as a single static frame),
# and a Smooth-PPA heatmap swept over the whole ground plane using that
# scenario's heading convention. All 4 heatmaps share one fixed color scale.
#
#   1. Facing the camera, in front of it
#   2. In front of the camera but turned to show more side profile
#   3. Back turned to the camera
#   4. On the edge of the camera's FOV, facing the camera

ENV["GKSwstype"] = "100"

if !@isdefined(ActorMeshTriangulated)
    include(joinpath(@__DIR__, "ActorMeshTriangulated.jl"))
end

using .ActorMeshTriangulated
using Plots
using LinearAlgebra

const OBJ_PATH = joinpath(@__DIR__, "simple_human_rotated_xaxis.obj")
const PART_DECAY = Dict(:Head_face => 2.0, :Body_face => 1.0, :Feet_face => 0.5, :Top_face => 1.5)

const DRONE_POS = [4.0, 0.0, 3.0]
const DRONE_YAW = Float64(pi)
const TILT = -0.35
const FOCAL_LENGTH = 1.2
const DRONE_VEC = [DRONE_POS[1], DRONE_POS[2], DRONE_POS[3], 0.0, 0.0, 0.0, DRONE_YAW, 0.0]
const FPV_VIEW_SIZE = 0.68

# Local stand-in for ActorTrajectory's ActorState, typed to the triangulated
# mesh instead of the box ActorMeshStruct.
struct TriActorState
    x::Float64
    y::Float64
    z::Float64
    heading::Float64
    mesh::TriMeshStruct
    actor_id::Int
end

# ── Camera projection (same convention as StaticPPA_OrigVsSmooth_FOV.jl) ─────
function project_point(v_world::Vector{Float64})
    dx = v_world[1] - DRONE_POS[1]
    dy = v_world[2] - DRONE_POS[2]
    dz = v_world[3] - DRONE_POS[3]
    bx = dx * cos(DRONE_YAW) + dy * sin(DRONE_YAW)
    by = -dx * sin(DRONE_YAW) + dy * cos(DRONE_YAW)
    cx_raw = bx * cos(TILT) + dz * sin(TILT)
    cz_raw = -bx * sin(TILT) + dz * cos(TILT)
    cx_denom = max(cx_raw, 0.1)
    u = FOCAL_LENGTH * by / cx_denom
    v_ = FOCAL_LENGTH * cz_raw / cx_denom
    return u, v_, cx_raw
end

# Like project_point, but returns `nothing` (instead of a value) when the
# point is behind the camera — used for line-drawing helpers (ground grid,
# axis lines) that should simply skip segments with an off-camera endpoint.
function project_or_nothing(v_world::Vector{Float64})
    u, v_, cx = project_point(v_world)
    cx < 0.05 && return nothing
    return (u, v_)
end

in_frame(u, v_, cx) = cx > 0.05 && abs(u) <= 1.0 && abs(v_) <= 1.0

function actor_center_uv(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64)
    return project_point([x, y, z + mesh.height / 2])
end

function whole_mesh_in_frame(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    for v in world_verts
        u, v_, cx = project_point(v)
        in_frame(u, v_, cx) || return false
    end
    return true
end

# Same scheme as Compositional_preferences_FPV.jl.
part_color(name::Symbol) = name == :Head_face ? :orange :
                           name == :Body_face ? :royalblue :
                           name == :Feet_face ? :seagreen : :crimson

# ── World View (3D) ───────────────────────────────────────────────────────────
function draw_quadcopter3d!(p, drone_state; arm_length=0.3, prop_radius=0.15)
    x, y, z, _, _, _, theta, _ = drone_state

    scatter!(p, [x], [y], [z], markersize=8, color=:darkred,
        markerstrokewidth=2, markerstrokecolor=:black, label="")

    for (i, arm_offset) in enumerate([π / 4, 3π / 4, 5π / 4, 7π / 4])
        angle = theta + arm_offset
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

    cam_x = x + 0.15 * cos(theta)
    cam_y = y + 0.15 * sin(theta)
    plot!(p, [x, cam_x], [y, cam_y], [z, z - 0.1], color=:lime, linewidth=3, label="")
    scatter!(p, [cam_x], [cam_y], [z - 0.1], markersize=3, color=:lime, marker=:square, label="")
end

function draw_actor_3d!(p, mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    for face in mesh.faces
        corners = [world_verts[idx] for idx in face.corner_indices]
        xs_f = [c[1] for c in corners]
        push!(xs_f, xs_f[1])
        ys_f = [c[2] for c in corners]
        push!(ys_f, ys_f[1])
        zs_f = [c[3] for c in corners]
        push!(zs_f, zs_f[1])
        # Plots.jl/GR does not support fillrange-based fill on 3D line series
        # (silently ignored), so color the triangle via its outline instead.
        plot!(p, xs_f, ys_f, zs_f, color=part_color(face.name), linewidth=2.0, label="")
    end
    arrow_len = 0.8
    hx = x + arrow_len * cos(heading)
    hy = y + arrow_len * sin(heading)
    plot!(p, [x, hx], [y, hy], [z, z], color=:green, linewidth=4, arrow=true, label="Heading")
end

# The 4 corners of the camera's rectangular FOV frustum at a fixed depth
# along its optical axis (derived directly from the same tilt/yaw/focal-length
# projection math as project_point, i.e. these corners project to exactly
# u,v = ±1 by construction).
# function fov_frustum_corners(depth::Float64)
#     f = FOCAL_LENGTH
#     corners = Vector{Float64}[]
#     for (su, sv) in [(-1,-1), (1,-1), (1,1), (-1,1)]
#         s = sv * depth / f
#         bx = depth*cos(TILT) - s*sin(TILT)
#         by = su * depth / f
#         dz = depth*sin(TILT) + s*cos(TILT)
#         dx = bx*cos(DRONE_YAW) - by*sin(DRONE_YAW)
#         dy = bx*sin(DRONE_YAW) + by*cos(DRONE_YAW)
#         push!(corners, [DRONE_POS[1]+dx, DRONE_POS[2]+dy, DRONE_POS[3]+dz])
#     end
#     return corners
# end

# function draw_fov_frustum!(p; depth::Float64 = 2.0)
#     corners = fov_frustum_corners(depth)
#     for c in corners
#         plot!(p, [DRONE_POS[1], c[1]], [DRONE_POS[2], c[2]], [DRONE_POS[3], c[3]],
#               color=:gray, linestyle=:dash, linewidth=1, alpha=0.6, label="")
#     end
#     xs_f = [c[1] for c in corners]; push!(xs_f, xs_f[1])
#     ys_f = [c[2] for c in corners]; push!(ys_f, ys_f[1])
#     zs_f = [c[3] for c in corners]; push!(zs_f, zs_f[1])
#     plot!(p, xs_f, ys_f, zs_f, color=:gray, linestyle=:dash, linewidth=1, alpha=0.6, label="")
# end

function build_world_view(mesh::TriMeshStruct, actor_x::Float64, actor_y::Float64, actor_z::Float64, heading::Float64)
    p_world = plot(
        xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="World View",
        legend=false, camera=(30, 45), aspect_ratio=:equal,
        xlims=(-8.0, 8.0), ylims=(-8.0, 8.0), zlims=(0.0, 5.0),
        background_color=:white
    )
    draw_actor_3d!(p_world, mesh, actor_x, actor_y, actor_z, heading)
    draw_quadcopter3d!(p_world, DRONE_VEC)
    #draw_fov_frustum!(p_world)
    mesh_center_z = actor_z + mesh.height / 2
    plot!(p_world, [DRONE_POS[1], actor_x], [DRONE_POS[2], actor_y], [DRONE_POS[3], mesh_center_z],
        linestyle=:dash, color=:gray, linewidth=1, alpha=0.6, label="")
    return p_world
end

# ── FPV Camera (black HUD-style view) ─────────────────────────────────────────
function draw_ground_grid_and_axes!(p; grid_range=-8:2:8)
    for gx in grid_range
        pa = project_or_nothing([Float64(gx), Float64(first(grid_range)), 0.0])
        pb = project_or_nothing([Float64(gx), Float64(last(grid_range)), 0.0])
        (pa === nothing || pb === nothing) || plot!(p, [pa[1], pb[1]], [pa[2], pb[2]], color=:gray40, linewidth=0.8, alpha=0.5, label="")
    end
    for gy in grid_range
        pa = project_or_nothing([Float64(first(grid_range)), Float64(gy), 0.0])
        pb = project_or_nothing([Float64(last(grid_range)), Float64(gy), 0.0])
        (pa === nothing || pb === nothing) || plot!(p, [pa[1], pb[1]], [pa[2], pb[2]], color=:gray40, linewidth=0.8, alpha=0.5, label="")
    end
    ax_len = Float64(last(grid_range))
    origin = project_or_nothing([0.0, 0.0, 0.0])
    for (tip, col) in [([ax_len, 0.0, 0.0], :red), ([0.0, ax_len, 0.0], :green), ([0.0, 0.0, ax_len], :dodgerblue)]
        pb = project_or_nothing(tip)
        (origin === nothing || pb === nothing) || plot!(p, [origin[1], pb[1]], [origin[2], pb[2]], color=col, linewidth=1.5, alpha=0.7, label="")
    end
end

function draw_fpv_actor!(p, mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    all_pts = Tuple{Float64,Float64}[]
    for face in mesh.faces
        proj = [project_or_nothing(world_verts[idx]) for idx in face.corner_indices]
        any(pt -> pt === nothing, proj) && continue
        us = [pt[1] for pt in proj]
        vs = [pt[2] for pt in proj]
        plot!(p, Shape(us, vs), fillalpha=0.65, fillcolor=part_color(face.name),
            linecolor=:black, linewidth=0.5, label="")
        append!(all_pts, zip(us, vs))
    end
    return all_pts
end

function draw_targeting_box!(p, pts::Vector{Tuple{Float64,Float64}})
    isempty(pts) && return
    us = first.(pts)
    vs = last.(pts)
    pad = 0.025
    u0, u1 = minimum(us) - pad, maximum(us) + pad
    v0, v1 = minimum(vs) - pad, maximum(vs) + pad
    plot!(p, [u0, u1, u1, u0, u0], [v0, v0, v1, v1, v0], color=:cyan, linewidth=1.6, linestyle=:dash, alpha=0.9, label="")
    annotate!(p, u0, v1 + 0.035, text("TARGET", :left, :cyan, 7))
end

function draw_hud_decorations!(p, actor_x, actor_y, actor_z, vs)
    # Crosshair
    ch, gap = 0.055, 0.015
    for (x1, x2, y1, y2) in [(gap, ch, 0.0, 0.0), (-ch, -gap, 0.0, 0.0), (0.0, 0.0, gap, ch), (0.0, 0.0, -ch, -gap)]
        plot!(p, [x1, x2], [y1, y2], color=:white, linewidth=2, alpha=0.9, label="")
    end
    scatter!(p, [0.0], [0.0], markersize=3, color=:white, markerstrokewidth=0, label="")

    # Corner bracket reticle
    bx_h, by_h, bl = 0.60, vs * 0.72 * 0.85, 0.07
    for (sx, sy) in [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        plot!(p, [sx * bx_h, sx * bx_h, sx * (bx_h - bl)], [sy * (by_h - bl), sy * by_h, sy * by_h],
            color=:white, linewidth=1.5, alpha=0.7, label="")
    end

    # HUD readouts
    dist = norm([actor_x - DRONE_POS[1], actor_y - DRONE_POS[2], actor_z - DRONE_POS[3]])
    tilt_deg = round(Int, TILT * 180 / π)
    xl = -vs
    annotate!(p, xl + 0.02, -vs * 0.72 + 0.06, text("DST  $(round(dist,digits=1))m", :left, :lime, 7))
    annotate!(p, xl + 0.02, -vs * 0.72 + 0.12, text("ALT  $(round(DRONE_POS[3],digits=1))m", :left, :lime, 7))
    annotate!(p, xl + 0.02, -vs * 0.72 + 0.18, text("TILT $(tilt_deg)°", :left, :dodgerblue, 7))
    annotate!(p, vs - 0.02, -vs * 0.72 + 0.05, text("FPV · DRONE CAM", :right, :white, 8))
    annotate!(p, vs - 0.02, vs * 0.72 - 0.04, text("● REC", :right, :red, 8))
end

function build_fpv_view(mesh::TriMeshStruct, actor_x::Float64, actor_y::Float64, actor_z::Float64, heading::Float64)
    vs = FPV_VIEW_SIZE
    p_fpv = plot(title="FPV Camera", legend=false,
        xlims=(-vs, vs), ylims=(-vs * 0.72, vs * 0.72), aspect_ratio=:equal,
        background_color=:black, foreground_color_axis=:white, foreground_color_border=:black,
        grid=false, ticks=false, framestyle=:box)
    draw_ground_grid_and_axes!(p_fpv)
    pts = draw_fpv_actor!(p_fpv, mesh, actor_x, actor_y, actor_z, heading)
    draw_targeting_box!(p_fpv, pts)
    draw_hud_decorations!(p_fpv, actor_x, actor_y, actor_z, vs)
    return p_fpv
end

# ── Smooth PPA (matches the current "require ALL vertices in FOV" rule in
# StaticPPA_OrigVsSmooth_FOV.jl) ──────────────────────────────────────────────
function shoelace_signed(pts::Vector)
    n = length(pts)
    A = 0.0
    for i in 1:n
        j = mod1(i + 1, n)
        A += pts[i][1] * pts[j][2] - pts[j][1] * pts[i][2]
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
        n_world = [face.normal[1] * c - face.normal[2] * s, face.normal[1] * s + face.normal[2] * c, face.normal[3]]
        weight = ActorMeshTriangulated.face_dynamic_weight(face, heading, actor_pos, drone_pos[1:2])

        uv = Vector{Vector{Float64}}()
        w = 1.0
        face_pos = sum(world_verts[idx] for idx in face.corner_indices) / length(face.corner_indices)
        dist = face_pos .- drone_pos
        n_dot = -dot(n_world, dist)
        is_occ = n_dot <= 0.0
        vis_n = 0
        for idx in face.corner_indices
            v = world_verts[idx]
            dx = v[1] - drone_pos[1]
            dy = v[2] - drone_pos[2]
            dz = v[3] - drone_pos[3]
            bx = dx * cos(drone_yaw) + dy * sin(drone_yaw)
            by = -dx * sin(drone_yaw) + dy * cos(drone_yaw)
            cx_raw = bx * cos(TILT) + dz * sin(TILT)
            cz_raw = -bx * sin(TILT) + dz * cos(TILT)
            cx_soft = (cx_raw + sqrt(cx_raw^2 + 1e-4)) / 2.0
            w *= cx_soft / (cx_soft + 0.1)
            cx_denom = max(cx_raw, 0.1)
            u = FOCAL_LENGTH * by / cx_denom
            v_ = FOCAL_LENGTH * cz_raw / cx_denom
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

# Same design as StaticPPA_OrigVsSmooth_FOV.jl's heatmaps: sweep every actor
# ground position, with heading recomputed at each point as "bearing to the
# fixed drone" plus this scenario's fixed offset (0/90/180°). The drone itself
# is fixed the whole time — only the actor position varies.
function compute_smooth_ppa_grid(mesh::TriMeshStruct, offset_deg::Float64; grid_size::Int=101)
    xs = range(-8.0, 8.0, length=grid_size)
    ys = range(-8.0, 8.0, length=grid_size)
    grid = [smooth_ppa_value(mesh, x, y, 0.0, atan(DRONE_POS[2] - y, DRONE_POS[1] - x) + deg2rad(offset_deg),
        DRONE_POS, DRONE_YAW)
            for y in ys, x in xs]
    return xs, ys, grid
end

# ── Search: closest-to-edge position where the actor is still fully in frame ──
function search_edge_of_fov(mesh::TriMeshStruct; xs=range(-7.0, 3.0, length=101), ys=range(-7.0, 7.0, length=141))
    best = (-Inf, 0.0, 0.0)
    for x in xs, y in ys
        heading = atan(DRONE_POS[2] - y, DRONE_POS[1] - x)
        whole_mesh_in_frame(mesh, x, y, 0.0, heading) || continue
        u, _, _ = actor_center_uv(mesh, x, y, 0.0)
        abs(u) > best[1] && (best = (abs(u), x, y))
    end
    return best[2], best[3]
end

# ── Search: position whose center projects closest to dead-center of frame ──
function search_center_position(mesh::TriMeshStruct; xs=range(-6.0, 3.0, length=91), ys=range(-6.0, 6.0, length=121))
    best = (Inf, 0.0, 0.0)
    for x in xs, y in ys
        u, v_, cx = actor_center_uv(mesh, x, y, 0.0)
        cx <= 0.05 && continue
        heading = atan(DRONE_POS[2] - y, DRONE_POS[1] - x)
        whole_mesh_in_frame(mesh, x, y, 0.0, heading) || continue
        d2 = u^2 + v_^2
        d2 < best[1] && (best = (d2, x, y))
    end
    return best[2], best[3]
end


# ── One experiment: World View + FPV Camera + Smooth PPA heatmap ────────────
# Heatmap panel matches StaticPPA_OrigVsSmooth_FOV.jl exactly: full sweep,
# fixed drone shown as a white square + white heading arrow, shared color
# scale across all 4 experiments — plus a marker + annotation at the actor's
# actual position showing the exact PPA value there (computed once directly).
function render_single_actor_figure(mesh::TriMeshStruct, title::String, out_file::String,
    actor_x::Float64, actor_y::Float64, heading::Float64,
    xs, ys, ppa_smooth, shared_max::Float64)
    p_world = build_world_view(mesh, actor_x, actor_y, 0.0, heading)
    p_fpv = build_fpv_view(mesh, actor_x, actor_y, 0.0, heading)

    ps_value = smooth_ppa_value(mesh, actor_x, actor_y, 0.0, heading, DRONE_POS, DRONE_YAW)

    p_heat = heatmap(xs, ys, ppa_smooth, c=:jet, clims=(0.0, shared_max), title="Smooth PPA",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box)
    scatter!(p_heat, [DRONE_POS[1]], [DRONE_POS[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="")
    quiver!(p_heat, [DRONE_POS[1]], [DRONE_POS[2]], quiver=([0.9 * cos(DRONE_YAW)], [0.9 * sin(DRONE_YAW)]), color=:white, lw=2.5)
    scatter!(p_heat, [actor_x], [actor_y], shape=:circle, ms=7, mc=:white, msc=:black, label="")
    quiver!(p_heat, [actor_x], [actor_y], quiver=([0.8 * cos(heading)], [0.8 * sin(heading)]), color=:white, lw=2.0)
    annotate!(p_heat, actor_x + 0.8, actor_y - 0.7, text("PPA=$(round(ps_value, digits=4))", :white, :left, 8))

    p_combined = plot(p_world, p_fpv, p_heat, layout=(1, 3), size=(1800, 600), plot_title=title)

    mkpath("src/mdma_greedy/drone_experiments")
    savefig(p_combined, out_file)
    println("✓ Saved → $out_file  (PPA=$(round(ps_value, digits=6)) at ($actor_x, $actor_y))")
    return p_combined
end

function run_all_single_actor_experiments()
    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)
    dir = "src/mdma_greedy/drone_experiments"

    ex, ey = search_edge_of_fov(mesh)
    cx, cy = search_center_position(mesh)

    scenarios = [
        ("Facing the Camera", "$dir/SingleActor_FacingCamera.png", cx, cy, 0.0),
        ("Side Profile Toward Camera", "$dir/SingleActor_SideProfile.png", cx, cy, 90.0),
        ("Back Turned to Camera", "$dir/SingleActor_BackTurned.png", cx, cy, 180.0),
        ("Edge of FOV, Facing Camera", "$dir/SingleActor_EdgeOfFOV.png", ex, ey, 0.0),
    ]

    # First pass: compute every heatmap grid (actor-position sweep) so we can
    # share one color scale, exactly like StaticPPA_OrigVsSmooth_FOV.jl does.
    grids = [compute_smooth_ppa_grid(mesh, offset_deg) for (_, _, _, _, offset_deg) in scenarios]
    shared_max = maximum(maximum(g[3]) for g in grids)
    println("Shared heatmap scale: 0.0 – $(round(shared_max, digits=4))")

    for ((title, out_file, actor_x, actor_y, offset_deg), (xs, ys, grid)) in zip(scenarios, grids)
        heading = atan(DRONE_POS[2] - actor_y, DRONE_POS[1] - actor_x) + deg2rad(offset_deg)
        render_single_actor_figure(mesh, title, out_file, actor_x, actor_y, heading, xs, ys, grid, shared_max)
    end
end

run_all_single_actor_experiments()
