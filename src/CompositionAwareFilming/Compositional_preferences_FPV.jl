# Compositional_preferences_FPV.jl
# Renders one example FPV camera frame per compositional preference defined in
# Compositional_preferences.jl: Rule of Thirds, Center Placement, Close Shot
# (whole head + part of body in frame), and Balance (one near subject, one
# far subject). For each, we search actor position(s) that satisfy that
# preference, then draw the actor's projected silhouette inside the camera
# frame with the relevant composition guides overlaid.

ENV["GKSwstype"] = "100"

if !@isdefined(ActorMeshTriangulated)
    include(joinpath(@__DIR__, "ActorMeshTriangulated.jl"))
end
if !@isdefined(CompositionalPreferences)
    include(joinpath(@__DIR__, "Compositional_preferences.jl"))
end

using .ActorMeshTriangulated
using .CompositionalPreferences
using Plots
using LinearAlgebra

const OBJ_PATH = joinpath(@__DIR__, "simple_human_rotated_xaxis.obj")
const PART_DECAY = Dict(:Head_face => 2.0, :Body_face => 1.0, :Feet_face => 0.5, :Top_face => 1.5)

const DRONE_POS = [4.0, 0.0, 2.0]
const DRONE_YAW = Float64(pi)
const TILT = -0.35
const FOCAL_LENGTH = 1.2

# Camera image half-extents in normalized image coords (u = f*side/depth,
# v = f*up/depth). The same bounds are used by the in-frame checks, the
# composition scores and the drawn frame. FOV = 2*atan(half / FOCAL_LENGTH).
const IMG_SQUARE = (w=1.0, h=1.0)
const IMG_16x9 = (w=1.0, h=0.5625)

# ── Camera projection (same convention as StaticPPA_OrigVsSmooth_FOV.jl) ─────
function project_point(v_world::Vector{Float64})
    dx = v_world[1] - DRONE_POS[1]
    dy = v_world[2] - DRONE_POS[2]
    dz = v_world[3] - DRONE_POS[3]
    bx = dx * cos(DRONE_YAW) + dy * sin(DRONE_YAW)
    by = -dx * sin(DRONE_YAW) + dy * cos(DRONE_YAW)
    depth = bx * cos(TILT) + dz * sin(TILT)
    cz_raw = -bx * sin(TILT) + dz * cos(TILT)
    depth_safe = max(depth, 0.1)
    u = FOCAL_LENGTH * by / depth_safe
    v_ = FOCAL_LENGTH * cz_raw / depth_safe
    return u, v_, depth
end

in_frame(u, v_, depth; img=IMG_16x9) = depth > 0.05 && abs(u) <= img.w && abs(v_) <= img.h

function actor_center_uv(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64)
    return project_point([x, y, z + mesh.height / 2])
end

# True only if EVERY vertex of the mesh (not just its center) projects inside
# the visible frame at this position/heading — the actual "nothing is clipped"
# constraint, as opposed to just checking the actor's center point.
function whole_mesh_in_frame(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64; img=IMG_16x9)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    for v in world_verts
        u, v_, depth = project_point(v)
        in_frame(u, v_, depth; img=img) || return false
    end
    return true
end

function project_mesh_triangles(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    tris = NamedTuple[]
    for face in mesh.faces
        pts = [project_point(world_verts[idx]) for idx in face.corner_indices]
        us = [p[1] for p in pts]
        vs = [p[2] for p in pts]
        depths = [p[3] for p in pts]
        push!(tris, (name=face.name, us=us, vs=vs, infront=all(d -> d > 0.05, depths)))
    end
    return tris
end

function head_uv_points(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    pts = Tuple{Float64,Float64}[]
    for face in mesh.faces
        face.name == :Head_face || continue
        for idx in face.corner_indices
            u, v_, depth = project_point(world_verts[idx])
            depth > 0.05 && push!(pts, (u, v_))
        end
    end
    return pts
end

part_color(name::Symbol) = name == :Head_face ? :orange :
                           name == :Body_face ? :royalblue :
                           name == :Feet_face ? :seagreen : :crimson

function draw_actor_silhouette!(p, tris)
    for t in tris
        t.infront || continue
        plot!(p, Shape(t.us, t.vs), fillalpha=0.65, linecolor=:black, linewidth=0.5, color=part_color(t.name), label="")
    end
end

# Draws the camera image border (±img.w × ±img.h) with a small margin around it.
function frame_axes!(p; img=IMG_16x9)
    pad = 0.1 * img.w
    plot!(p, xlims=(-img.w - pad, img.w + pad), ylims=(-img.h - pad, img.h + pad), aspect_ratio=:equal,
        grid=false, ticks=false, framestyle=:box, legend=false)
    plot!(p, [-img.w, img.w, img.w, -img.w, -img.w], [-img.h, -img.h, img.h, img.h, -img.h], color=:black, linewidth=1.5)
end

# Hides everything drawn outside the camera image: white bands over the margin
# around the frame (large enough to cover anything projected far outside),
# then the frame border on top, so only what the camera captures is visible.
function mask_outside_frame!(p; img=IMG_16x9)
    W, H, M = img.w, img.h, 100.0
    for (xs, ys) in (([-M, M, M, -M], [H, H, M, M]),          # above the frame
        ([-M, M, M, -M], [-M, -M, -H, -H]),      # below
        ([-M, -W, -W, -M], [-H, -H, H, H]),      # left
        ([W, M, M, W], [-H, -H, H, H]))          # right
        plot!(p, Shape(xs, ys), fillcolor=:white, linecolor=:white, linewidth=0, label="")
    end
    plot!(p, [-W, W, W, -W, -W], [-H, -H, H, H, -H], color=:black, linewidth=1.5, label="")
    pad = 0.1 * W
    plot!(p, xlims=(-W - pad, W + pad), ylims=(-H - pad, H + pad))   # keep the view limits
end

# ── Brute-force search over actor ground position, scoring the actor's center point ──
function search_best_center(score_fn; xs=range(-6.0, 3.0, length=91), ys=range(-6.0, 6.0, length=121), z=0.0, mesh,
    img=IMG_16x9)
    best = (-Inf, 0.0, 0.0)
    for x in xs, y in ys
        u, v_, depth = actor_center_uv(mesh, x, y, z)
        depth <= 0.05 && continue
        heading = atan(DRONE_POS[2] - y, DRONE_POS[1] - x)
        whole_mesh_in_frame(mesh, x, y, z, heading; img=img) || continue
        s = score_fn(u, v_)
        s > best[1] && (best = (s, x, y))
    end
    return best
end

# Closest position where the WHOLE head is inside the frame and at least part
# of the body is also visible — i.e. a classic head-and-shoulders close shot,
# rather than maximizing raw head-only frame coverage.
function search_close_shot_full_head(mesh; xs=range(0.5, 3.95, length=139), ys=range(-2.5, 2.5, length=101), z=0.0,
    img=IMG_16x9)
    best_dist = Inf
    best_xy = (0.0, 0.0)
    for x in xs, y in ys
        heading = atan(DRONE_POS[2] - y, DRONE_POS[1] - x)
        world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)

        head_ok = true
        for face in mesh.faces
            face.name == :Head_face || continue
            for idx in face.corner_indices
                u, v_, depth = project_point(world_verts[idx])
                in_frame(u, v_, depth; img=img) || (head_ok = false)
                head_ok || break
            end
            head_ok || break
        end
        head_ok || continue

        body_ok = false
        for face in mesh.faces
            face.name == :Body_face || continue
            for idx in face.corner_indices
                u, v_, depth = project_point(world_verts[idx])
                in_frame(u, v_, depth; img=img) && (body_ok = true)
                body_ok && break
            end
            body_ok && break
        end
        body_ok || continue

        d = norm([x, y, z] .- DRONE_POS)
        if d < best_dist
            best_dist = d
            best_xy = (x, y)
        end
    end
    return best_xy
end

# One near actor at a fixed off-center offset, then search a far actor's
# position (on the opposite side) that best balances the weighted centroid —
# weight ~ 1/distance, so the closer (visually larger) subject counts more,
# matching the classic near-large / far-small composition balance.
function search_balance_near_far(mesh; near_xy=(1.5, 1.5), far_xs=range(-5.0, 3.5, length=191), far_ys=range(-5.0, 5.0, length=241),
    img=IMG_16x9)
    x_near, y_near = near_xy
    u_near, v_near, _ = actor_center_uv(mesh, x_near, y_near, 0.0)
    d_near = norm([x_near, y_near, 0.0] .- DRONE_POS)
    w_near = 1.0 / d_near

    best = (-Inf, 0.0, 0.0)
    for x in far_xs, y in far_ys
        u_far, v_far, depth_far = actor_center_uv(mesh, x, y, 0.0)
        in_frame(u_far, v_far, depth_far; img=img) || continue
        heading_far = atan(DRONE_POS[2] - y, DRONE_POS[1] - x)
        whole_mesh_in_frame(mesh, x, y, 0.0, heading_far; img=img) || continue
        d_far = norm([x, y, 0.0] .- DRONE_POS)
        d_far > d_near || continue   # must actually be farther than the "near" actor
        w_far = 1.0 / d_far
        s = CompositionalPreferences.balance_score([(u_near, v_near), (u_far, v_far)], [w_near, w_far])
        s > best[1] && (best = (s, x, y))
    end
    return best, (u_near, v_near, w_near)
end

# `img` is the camera image (IMG_16x9 or IMG_SQUARE); `tag` is appended to the
# output filename so different image shapes don't overwrite each other.
function run_compositional_preferences_fpv(; img=IMG_16x9, tag::String="")
    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)
    W, H = img.w, img.h

    # ── 1. Rule of Thirds ──────────────────────────────────────────────────
    s1, x1, y1 = search_best_center((u, v) -> CompositionalPreferences.rule_of_thirds_score(u, v; img=img);
        mesh=mesh, img=img)
    heading1 = atan(DRONE_POS[2] - y1, DRONE_POS[1] - x1)
    u1, v1, _ = actor_center_uv(mesh, x1, y1, 0.0)
    p1 = plot(title="(1) Rule of Thirds")
    frame_axes!(p1; img=img)
    plot!(p1, [-W, W], [-H / 3, -H / 3], color=:gray, linestyle=:dash, linewidth=1)
    plot!(p1, [-W, W], [H / 3, H / 3], color=:gray, linestyle=:dash, linewidth=1)
    plot!(p1, [-W / 3, -W / 3], [-H, H], color=:gray, linestyle=:dash, linewidth=1)
    plot!(p1, [W / 3, W / 3], [-H, H], color=:gray, linestyle=:dash, linewidth=1)
    draw_actor_silhouette!(p1, project_mesh_triangles(mesh, x1, y1, 0.0, heading1))
    mask_outside_frame!(p1; img=img)

    # ── 2. Center Placement ────────────────────────────────────────────────
    s2, x2, y2 = search_best_center(CompositionalPreferences.center_placement_score; mesh=mesh, img=img)
    heading2 = atan(DRONE_POS[2] - y2, DRONE_POS[1] - x2)
    u2, v2, _ = actor_center_uv(mesh, x2, y2, 0.0)
    p2 = plot(title="(2) Center Placement")
    frame_axes!(p2; img=img)
    plot!(p2, [-W, W], [0, 0], color=:gray, linestyle=:dash, linewidth=1)
    plot!(p2, [0, 0], [-H, H], color=:gray, linestyle=:dash, linewidth=1)
    draw_actor_silhouette!(p2, project_mesh_triangles(mesh, x2, y2, 0.0, heading2))
    mask_outside_frame!(p2; img=img)

    # ── 3. Close Shot (whole head + part of body) ────────────────────────────
    x3, y3 = search_close_shot_full_head(mesh; img=img)
    heading3 = atan(DRONE_POS[2] - y3, DRONE_POS[1] - x3)
    head_pts = head_uv_points(mesh, x3, y3, 0.0, heading3)
    p3 = plot(title="(3) Close Shot — Head")
    frame_axes!(p3; img=img)
    draw_actor_silhouette!(p3, project_mesh_triangles(mesh, x3, y3, 0.0, heading3))
    mask_outside_frame!(p3; img=img)      # parts of the actor outside the frame are not shown
    if !isempty(head_pts)
        us = first.(head_pts)
        vs = last.(head_pts)
        plot!(p3, [minimum(us), maximum(us), maximum(us), minimum(us), minimum(us)],
            [minimum(vs), minimum(vs), maximum(vs), maximum(vs), minimum(vs)],
            color=:red, linewidth=2, linestyle=:dash)
    end

    # ── 4. Balance — one subject close, one far ──────────────────────────────
    (s4, xf, yf), (u_near, v_near, w_near) = search_balance_near_far(mesh; img=img)
    x_near, y_near = 1.5, 1.5
    headingA = atan(DRONE_POS[2] - y_near, DRONE_POS[1] - x_near)
    headingB = atan(DRONE_POS[2] - yf, DRONE_POS[1] - xf)
    u_far, v_far, _ = actor_center_uv(mesh, xf, yf, 0.0)
    w_far = 1.0 / norm([xf, yf, 0.0] .- DRONE_POS)
    centroid_u = (w_near * u_near + w_far * u_far) / (w_near + w_far)
    centroid_v = (w_near * v_near + w_far * v_far) / (w_near + w_far)
    p4 = plot(title="(4) Balance — Near + Far Subject")
    frame_axes!(p4; img=img)
    plot!(p4, [0, 0], [-H, H], color=:gray, linestyle=:dash, linewidth=1)
    draw_actor_silhouette!(p4, project_mesh_triangles(mesh, x_near, y_near, 0.0, headingA))
    draw_actor_silhouette!(p4, project_mesh_triangles(mesh, xf, yf, 0.0, headingB))
    mask_outside_frame!(p4; img=img)

    p_combined = plot(p1, p2, p3, p4, layout=(2, 2), size=(1100, 1100),
        plot_title="Compositional Preferences — First Person View Examples")

    mkpath("src/CompositionAwareFilming/drone_experiments")
    out_file = "src/CompositionAwareFilming/drone_experiments/Compositional_Preferences_FPV$tag.png"
    savefig(p_combined, out_file)
    println("✓ Saved → $out_file")
    println("  (1) Rule of Thirds : score=$(round(s1,digits=4)) at (x=$x1, y=$y1)")
    println("  (2) Center         : score=$(round(s2,digits=4)) at (x=$x2, y=$y2)")
    println("  (3) Close Shot     : whole head + body visible at (x=$x3, y=$y3)")
    println("  (4) Balance        : score=$(round(s4,digits=4)); near=($x_near,$y_near) far=($xf,$yf)")

    return p_combined
end

run_compositional_preferences_fpv_16x9() = run_compositional_preferences_fpv(img=IMG_16x9, tag="_16x9")
run_compositional_preferences_fpv_square() = run_compositional_preferences_fpv(img=IMG_SQUARE, tag="_Square")

run_compositional_preferences_fpv_16x9()
run_compositional_preferences_fpv_square()
