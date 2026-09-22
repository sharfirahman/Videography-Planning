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

const OBJ_PATH = joinpath(@__DIR__, "simple_human_rotated_color.obj")
const PART_DECAY = Dict(:Head_face => 2.0, :Body_face => 1.0, :Feet_face => 0.5, :Top_face => 1.5)

const DRONE_POS = [4.0, 0.0, 2.0]
const DRONE_YAW = Float64(pi)
const TILT = -0.35
const FOCAL_LENGTH = 1.2

# ── Camera projection (same convention as StaticPPA_OrigVsSmooth_FOV.jl) ─────
function project_point(v_world::Vector{Float64})
    dx = v_world[1] - DRONE_POS[1]; dy = v_world[2] - DRONE_POS[2]; dz = v_world[3] - DRONE_POS[3]
    bx = dx*cos(DRONE_YAW) + dy*sin(DRONE_YAW)
    by = -dx*sin(DRONE_YAW) + dy*cos(DRONE_YAW)
    cx_raw = bx*cos(TILT) + dz*sin(TILT)
    cz_raw = -bx*sin(TILT) + dz*cos(TILT)
    cx_denom = max(cx_raw, 0.1)
    u = FOCAL_LENGTH * by / cx_denom
    v_ = FOCAL_LENGTH * cz_raw / cx_denom
    return u, v_, cx_raw
end

in_frame(u, v_, cx) = cx > 0.05 && abs(u) <= 1.0 && abs(v_) <= 1.0

function actor_center_uv(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64)
    return project_point([x, y, z + mesh.height/2])
end

# True only if EVERY vertex of the mesh (not just its center) projects inside
# the visible frame at this position/heading — the actual "nothing is clipped"
# constraint, as opposed to just checking the actor's center point.
function whole_mesh_in_frame(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    for v in world_verts
        u, v_, cx = project_point(v)
        in_frame(u, v_, cx) || return false
    end
    return true
end

function project_mesh_triangles(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    tris = NamedTuple[]
    for face in mesh.faces
        pts = [project_point(world_verts[idx]) for idx in face.corner_indices]
        us = [p[1] for p in pts]; vs = [p[2] for p in pts]; cxs = [p[3] for p in pts]
        push!(tris, (name = face.name, us = us, vs = vs, infront = all(c -> c > 0.05, cxs)))
    end
    return tris
end

function head_uv_points(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    pts = Tuple{Float64,Float64}[]
    for face in mesh.faces
        face.name == :Head_face || continue
        for idx in face.corner_indices
            u, v_, cx = project_point(world_verts[idx])
            cx > 0.05 && push!(pts, (u, v_))
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

function frame_axes!(p)
    plot!(p, xlims=(-1.1, 1.1), ylims=(-1.1, 1.1), aspect_ratio=:equal,
        grid=false, ticks=false, framestyle=:box, legend=false)
    plot!(p, [-1,1,1,-1,-1], [-1,-1,1,1,-1], color=:black, linewidth=1.5)
end

# ── Brute-force search over actor ground position, scoring the actor's center point ──
function search_best_center(score_fn; xs=range(-6.0, 3.0, length=91), ys=range(-6.0, 6.0, length=121), z=0.0, mesh)
    best = (-Inf, 0.0, 0.0)
    for x in xs, y in ys
        u, v_, cx = actor_center_uv(mesh, x, y, z)
        cx <= 0.05 && continue
        heading = atan(DRONE_POS[2]-y, DRONE_POS[1]-x)
        whole_mesh_in_frame(mesh, x, y, z, heading) || continue
        s = score_fn(u, v_)
        s > best[1] && (best = (s, x, y))
    end
    return best
end

# Closest position where the WHOLE head is inside the frame and at least part
# of the body is also visible — i.e. a classic head-and-shoulders close shot,
# rather than maximizing raw head-only frame coverage.
function search_close_shot_full_head(mesh; xs=range(0.5, 3.95, length=139), ys=range(-2.5, 2.5, length=101), z=0.0)
    best_dist = Inf
    best_xy = (0.0, 0.0)
    for x in xs, y in ys
        heading = atan(DRONE_POS[2]-y, DRONE_POS[1]-x)
        world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)

        head_ok = true
        for face in mesh.faces
            face.name == :Head_face || continue
            for idx in face.corner_indices
                u, v_, cx = project_point(world_verts[idx])
                in_frame(u, v_, cx) || (head_ok = false)
                head_ok || break
            end
            head_ok || break
        end
        head_ok || continue

        body_ok = false
        for face in mesh.faces
            face.name == :Body_face || continue
            for idx in face.corner_indices
                u, v_, cx = project_point(world_verts[idx])
                in_frame(u, v_, cx) && (body_ok = true)
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
function search_balance_near_far(mesh; near_xy=(1.5, 1.5), far_xs=range(-5.0, 3.5, length=191), far_ys=range(-5.0, 5.0, length=241))
    x_near, y_near = near_xy
    u_near, v_near, _ = actor_center_uv(mesh, x_near, y_near, 0.0)
    d_near = norm([x_near, y_near, 0.0] .- DRONE_POS)
    w_near = 1.0 / d_near

    best = (-Inf, 0.0, 0.0)
    for x in far_xs, y in far_ys
        u_far, v_far, cx_far = actor_center_uv(mesh, x, y, 0.0)
        in_frame(u_far, v_far, cx_far) || continue
        heading_far = atan(DRONE_POS[2]-y, DRONE_POS[1]-x)
        whole_mesh_in_frame(mesh, x, y, 0.0, heading_far) || continue
        d_far = norm([x, y, 0.0] .- DRONE_POS)
        d_far > d_near || continue   # must actually be farther than the "near" actor
        w_far = 1.0 / d_far
        s = CompositionalPreferences.balance_score([(u_near,v_near), (u_far,v_far)], [w_near, w_far])
        s > best[1] && (best = (s, x, y))
    end
    return best, (u_near, v_near, w_near)
end

function run_compositional_preferences_fpv()
    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)

    # ── 1. Rule of Thirds ──────────────────────────────────────────────────
    s1, x1, y1 = search_best_center(CompositionalPreferences.rule_of_thirds_score; mesh=mesh)
    heading1 = atan(DRONE_POS[2]-y1, DRONE_POS[1]-x1)
    u1, v1, _ = actor_center_uv(mesh, x1, y1, 0.0)
    p1 = plot(title="Rule of Thirds")
    frame_axes!(p1)
    plot!(p1, [-1,1], [-1/3,-1/3], color=:gray, linestyle=:dash, linewidth=1)
    plot!(p1, [-1,1], [1/3,1/3], color=:gray, linestyle=:dash, linewidth=1)
    plot!(p1, [-1/3,-1/3], [-1,1], color=:gray, linestyle=:dash, linewidth=1)
    plot!(p1, [1/3,1/3], [-1,1], color=:gray, linestyle=:dash, linewidth=1)
    draw_actor_silhouette!(p1, project_mesh_triangles(mesh, x1, y1, 0.0, heading1))

    # ── 2. Center Placement ────────────────────────────────────────────────
    s2, x2, y2 = search_best_center(CompositionalPreferences.center_placement_score; mesh=mesh)
    heading2 = atan(DRONE_POS[2]-y2, DRONE_POS[1]-x2)
    u2, v2, _ = actor_center_uv(mesh, x2, y2, 0.0)
    p2 = plot(title="Center Placement")
    frame_axes!(p2)
    plot!(p2, [-1,1], [0,0], color=:gray, linestyle=:dash, linewidth=1)
    plot!(p2, [0,0], [-1,1], color=:gray, linestyle=:dash, linewidth=1)
    draw_actor_silhouette!(p2, project_mesh_triangles(mesh, x2, y2, 0.0, heading2))

    # ── 3. Close Shot (whole head + part of body) ────────────────────────────
    x3, y3 = search_close_shot_full_head(mesh)
    heading3 = atan(DRONE_POS[2]-y3, DRONE_POS[1]-x3)
    head_pts = head_uv_points(mesh, x3, y3, 0.0, heading3)
    p3 = plot(title="Close Shot — Head + Body")
    frame_axes!(p3)
    draw_actor_silhouette!(p3, project_mesh_triangles(mesh, x3, y3, 0.0, heading3))
    if !isempty(head_pts)
        us = first.(head_pts); vs = last.(head_pts)
        plot!(p3, [minimum(us),maximum(us),maximum(us),minimum(us),minimum(us)],
                  [minimum(vs),minimum(vs),maximum(vs),maximum(vs),minimum(vs)],
                  color=:red, linewidth=2, linestyle=:dash)
    end

    # ── 4. Balance — one subject close, one far ──────────────────────────────
    (s4, xf, yf), (u_near, v_near, w_near) = search_balance_near_far(mesh)
    x_near, y_near = 1.5, 1.5
    headingA = atan(DRONE_POS[2]-y_near, DRONE_POS[1]-x_near)
    headingB = atan(DRONE_POS[2]-yf, DRONE_POS[1]-xf)
    u_far, v_far, _ = actor_center_uv(mesh, xf, yf, 0.0)
    w_far = 1.0 / norm([xf, yf, 0.0] .- DRONE_POS)
    centroid_u = (w_near*u_near + w_far*u_far) / (w_near + w_far)
    centroid_v = (w_near*v_near + w_far*v_far) / (w_near + w_far)
    p4 = plot(title="Balance — Near + Far Subject")
    frame_axes!(p4)
    plot!(p4, [0,0], [-1,1], color=:gray, linestyle=:dash, linewidth=1)
    draw_actor_silhouette!(p4, project_mesh_triangles(mesh, x_near, y_near, 0.0, headingA))
    draw_actor_silhouette!(p4, project_mesh_triangles(mesh, xf, yf, 0.0, headingB))

    p_combined = plot(p1, p2, p3, p4, layout=(2,2), size=(1100,1100),
        plot_title="Compositional Preferences — FPV Examples")

    mkpath("src/mdma_greedy/drone_experiments")
    out_file = "src/mdma_greedy/drone_experiments/Compositional_Preferences_FPV.png"
    savefig(p_combined, out_file)
    println("✓ Saved → $out_file")
    println("  Rule of Thirds : score=$(round(s1,digits=4)) at (x=$x1, y=$y1)")
    println("  Center         : score=$(round(s2,digits=4)) at (x=$x2, y=$y2)")
    println("  Close Shot     : whole head + body visible at (x=$x3, y=$y3)")
    println("  Balance        : score=$(round(s4,digits=4)); near=($x_near,$y_near) far=($xf,$yf)")

    return p_combined
end

run_compositional_preferences_fpv()
