# PPA_QuadVsTriangulated.jl
# Compares PPA values computed from the human mesh's original (quad) faces,
# read directly from the .obj file, against the same mesh loaded through
# GLMakie/MeshIO (which triangulates on load). Same drone/actor pose is used
# for both, so any difference in PPA comes purely from face triangulation.

ENV["GKSwstype"] = "100"

if !@isdefined(ActorMesh)
    include(joinpath(@__DIR__, "ActorMesh.jl"))
end
if !@isdefined(ActorTrajectory)
    include(joinpath(@__DIR__, "ActorTrajectory.jl"))
end

using .ActorMesh
using .ActorTrajectory
import GLMakie: assetpath
using FileIO: load
using GeometryBasics
using LinearAlgebra
using Plots
using Base.Iterators

const OBJ_PATH = joinpath(@__DIR__, "simple_human_rotated_with_texture.obj")

# ── Raw .obj parsing (keeps original face degree, no triangulation) ───────────
function parse_obj_quads(path::String)
    vertices = Vector{Vector{Float64}}()
    face_index_lists = Vector{Vector{Int}}()

    for line in eachline(path)
        parts = split(strip(line))
        isempty(parts) && continue

        if parts[1] == "v"
            push!(vertices, parse.(Float64, parts[2:4]))
        elseif parts[1] == "f"
            idxs = [parse(Int, split(tok, "/")[1]) for tok in parts[2:end]]
            push!(face_index_lists, idxs)
        end
    end

    return vertices, face_index_lists
end

# ── MeshIO-loaded mesh (triangulated on load) ──────────────────────────────────
function load_obj_triangles(path::String)
    m = load(assetpath(path))
    verts = [Vector{Float64}(v) for v in GeometryBasics.coordinates(m)]
    fcs = [Vector{Int}([Int(Base.to_index(idx)) for idx in f]) for f in GeometryBasics.faces(m)]
    return verts, fcs
end

# ── Generic planar-polygon normal / area / centroid (Newell's method) ─────────
function polygon_normal_area_centroid(corners::Vector{Vector{Float64}})
    n = length(corners)
    centroid = sum(corners) / n

    normal_sum = zeros(3)
    for j in 1:n
        v1 = corners[j]
        v2 = corners[mod1(j + 1, n)]
        normal_sum += cross(v1 - centroid, v2 - centroid)
    end
    area = norm(normal_sum) / 2
    normal = normal_sum / norm(normal_sum)

    return normal, area, centroid
end

# ── Build an ActorMeshStruct from arbitrary vertices/face index lists ─────────
function build_actor_mesh_from_faces(vertices::Vector{Vector{Float64}}, face_index_lists::Vector{Vector{Int}}; weight::Float64=1.0)
    faces = ActorFace[]
    for (i, idxs) in enumerate(face_index_lists)
        corners = vertices[idxs]
        normal, area, centroid = polygon_normal_area_centroid(corners)
        push!(faces, ActorFace(Symbol("f$i"), weight, normal, centroid, :gray, area, idxs))
    end

    zs = [v[3] for v in vertices]
    xs = [v[1] for v in vertices]
    ys = [v[2] for v in vertices]
    height = maximum(zs) - minimum(zs)
    depth  = maximum(xs) - minimum(xs)
    width  = maximum(ys) - minimum(ys)

    return ActorMeshStruct(vertices, faces, Tuple{Int,Int}[], width, depth, height)
end

# ── PPA functions (duplicated from StaticPPA_OrigVsSmooth_FOV.jl) ─────────────
function shoelace_signed_static(pts::Vector)
    n = length(pts)
    A = 0.0
    for i in 1:n
        j = mod1(i + 1, n)
        A += pts[i][1] * pts[j][2] - pts[j][1] * pts[i][2]
    end
    return A / 2.0
end

function isvisible_hard_static(dist::Vector{Float64}, face_normal::Vector{Float64})
    -dot(dist, face_normal) > 0.0 ? 1.0 : 0.0
end

function ppa_coverage_original_static(
    face::ActorFace,
    face_pos::Vector{Float64},
    n_world::Vector{Float64},
    drone_pos::Vector{Float64},
    drone_yaw::Float64
)
    alpha = 1
    dist    = face_pos .- drone_pos
    d4      = norm(dist)^4
    d4 < 1e-6 && return 0.0
    heading = [cos(drone_yaw), sin(drone_yaw), 0.0]

    pixel_density = alpha * abs(dot(dist, heading)) *
           (-dot(dist, n_world))  *
           isvisible_hard_static(dist, n_world) / d4

    return pixel_density
end

function ppa_quality_original_static(face::ActorFace, cov::Float64)
    face.weight * face.area * sqrt(max(cov, 0.0))
end

function ppa_coverage_smooth_static(
    face::ActorFace,
    world_verts::Vector,
    drone::Vector{Float64},
    n_world::Vector{Float64};
    focal_length::Float64 = 1.2,
    tilt::Float64         = -0.35
)
    yaw = drone[7]
    uv  = Vector{Vector{Float64}}()
    w   = 1.0

    face_pos = sum(world_verts[idx] for idx in face.corner_indices) / length(face.corner_indices)
    dist     = face_pos .- drone[1:3]
    n_dot    = -dot(n_world, dist)

    is_occluded = n_dot <= 0.0

    visible_vertices = 0
    n_corners = length(face.corner_indices)

    for idx in face.corner_indices
        v  = world_verts[idx]
        dx = v[1] - drone[1];  dy = v[2] - drone[2];  dz = v[3] - drone[3]
        bx =  dx * cos(yaw) + dy * sin(yaw)
        by = -dx * sin(yaw) + dy * cos(yaw)
        cx_raw =  bx * cos(tilt) + dz * sin(tilt)
        cy     =  by
        cz_raw = -bx * sin(tilt) + dz * cos(tilt)

        cx_soft = (cx_raw + sqrt(cx_raw^2 + 1e-4)) / 2.0
        w *= cx_soft / (cx_soft + 0.1)

        cx_denom = max(cx_raw, 0.1)
        u  = focal_length * cy     / cx_denom
        v_ = focal_length * cz_raw / cx_denom
        push!(uv, [u, v_])

        if !is_occluded && cx_raw > 0.05 && abs(u) <= 1.0 && abs(v_) <= 1.0
            visible_vertices += 1
        end
    end

    # Same "at least half the corners visible" rule as the quad-based
    # original, generalized to n_corners instead of a hardcoded 4.
    w *= visible_vertices >= ceil(n_corners / 2) ? 1.0 : 0.0

    vis_soft = (n_dot + sqrt(n_dot^2 + 1e-4)) / 2.0

    A_mag = abs(shoelace_signed_static(uv))
    return w * vis_soft * A_mag
end

function ppa_quality_smooth_static(face::ActorFace, cov::Float64)
    face.weight * cov
end

function eval_ppa_static(
    actor::ActorState,
    drone::Vector{Float64};
    focal_length::Float64 = 1.2,
    tilt::Float64         = -0.35
)
    drone_pos = drone[1:3]
    drone_yaw = drone[7]

    po = 0.0
    for face in actor.mesh.faces
        face_pos  = actor_world_face_center(actor.mesh, face, actor.x, actor.y, actor.z, actor.heading)
        n_world   = actor_world_normal(face, actor.heading)
        cov_o     = ppa_coverage_original_static(face, face_pos, n_world, drone_pos, drone_yaw)
        po       += ppa_quality_original_static(face, cov_o)
    end

    world_verts = actor_world_vertices(actor.mesh, actor.x, actor.y, actor.z, actor.heading)
    ps = 0.0
    for face in actor.mesh.faces
        n_world = actor_world_normal(face, actor.heading)
        ps += ppa_quality_smooth_static(face,
                 ppa_coverage_smooth_static(face, world_verts, drone, n_world;
                                     focal_length=focal_length, tilt=tilt))
    end
    return po, ps
end

# ── Main comparison ────────────────────────────────────────────────────────────
function compare_ppa_quad_vs_triangulated(; verbose::Bool=false)
    println("="^70)
    println("PPA Comparison: Original Quad Faces vs. Triangulated Faces")
    println("="^70)

    quad_verts, quad_face_idxs = parse_obj_quads(OBJ_PATH)
    tri_verts,  tri_face_idxs  = load_obj_triangles(OBJ_PATH)

    println("  Quad mesh        : $(length(quad_verts)) vertices, $(length(quad_face_idxs)) faces")
    println("  Triangulated mesh: $(length(tri_verts)) vertices, $(length(tri_face_idxs)) faces")
    println()

    mesh_quad = build_actor_mesh_from_faces(quad_verts, quad_face_idxs)
    mesh_tri  = build_actor_mesh_from_faces(tri_verts, tri_face_idxs)

    # Fixed drone/actor pose — same for both meshes.
    drone_pos = [4.0, 0.0, 2.0]
    drone_yaw = Float64(pi)
    drone_vec = [drone_pos[1], drone_pos[2], drone_pos[3], 0.0, 0.0, 0.0, drone_yaw, 0.0]

    actor_x, actor_y, actor_z = -2.0, 0.0, 0.0
    actor_heading = atan(drone_pos[2] - actor_y, drone_pos[1] - actor_x)

    println("  Actor Position : ($actor_x, $actor_y, $actor_z)")
    println("  Actor Heading  : $(round(rad2deg(actor_heading), digits=2))°")
    println("  Drone Position : ($(drone_pos[1]), $(drone_pos[2]), $(drone_pos[3]))")
    println("  Drone Yaw      : $(round(rad2deg(drone_yaw), digits=2))°")
    println()

    actor_quad = ActorState(actor_x, actor_y, actor_z, actor_heading, mesh_quad, 1)
    actor_tri  = ActorState(actor_x, actor_y, actor_z, actor_heading, mesh_tri, 1)

    po_quad, ps_quad = eval_ppa_static(actor_quad, drone_vec)
    po_tri,  ps_tri  = eval_ppa_static(actor_tri, drone_vec)

    println("  Original PPA — quad faces        : $(round(po_quad, digits=6))")
    println("  Original PPA — triangulated faces: $(round(po_tri, digits=6))")
    println("  Difference                       : $(round(po_tri - po_quad, digits=6)) ($(round(100*(po_tri-po_quad)/max(po_quad,1e-9), digits=2))%)")
    println()
    println("  Smooth PPA — quad faces          : $(round(ps_quad, digits=6))")
    println("  Smooth PPA — triangulated faces  : $(round(ps_tri, digits=6))")
    println("  Difference                       : $(round(ps_tri - ps_quad, digits=6)) ($(round(100*(ps_tri-ps_quad)/max(ps_quad,1e-9), digits=2))%)")

    if verbose
        println()
        println("  Per-face breakdown (quad mesh):")
        world_verts_q = actor_world_vertices(mesh_quad, actor_x, actor_y, actor_z, actor_heading)
        for face in mesh_quad.faces
            n_world = actor_world_normal(face, actor_heading)
            cov = ppa_coverage_smooth_static(face, world_verts_q, drone_vec, n_world)
            println("    $(face.name): area=$(round(face.area,digits=4)) smooth_cov=$(round(cov,digits=6))")
        end
    end

    return po_quad, ps_quad, po_tri, ps_tri
end

compare_ppa_quad_vs_triangulated(verbose=true)

# ── Grid heatmap: Quad mesh vs. Triangulated mesh ──────────────────────────────
# Same setup as Experiment 1 in StaticPPA_OrigVsSmooth_FOV.jl (actor sweeps the
# full 2D grid, heading toward the drone at every point; drone fixed at
# (4,0,2), yaw=π) — but instead of Original-vs-Smooth, this compares the two
# meshes (quad faces vs. MeshIO-triangulated faces) for both PPA variants.
function run_static_comparison_quad_vs_tri(; global_max::Union{Nothing,Float64}=nothing)
    println("\n" * "="^70)
    println("Grid Comparison: Quad-Face Mesh vs. Triangulated Mesh")
    println("="^70)

    mkpath(joinpath(@__DIR__, "drone_experiments"))

    quad_verts, quad_face_idxs = parse_obj_quads(OBJ_PATH)
    tri_verts,  tri_face_idxs  = load_obj_triangles(OBJ_PATH)
    mesh_quad = build_actor_mesh_from_faces(quad_verts, quad_face_idxs)
    mesh_tri  = build_actor_mesh_from_faces(tri_verts, tri_face_idxs)

    drone_pos = [4.0, 0.0, 2.0]
    drone_yaw = Float64(pi)
    drone_vec = [drone_pos[1], drone_pos[2], drone_pos[3], 0.0, 0.0, 0.0, drone_yaw, 0.0]

    actor_z = 0.0

    println("  Drone Position : ($(drone_pos[1]), $(drone_pos[2]), $(drone_pos[3]))")
    println("  Drone Yaw      : $(round(rad2deg(drone_yaw), digits=2))°")

    grid_size = 201
    xs = range(-8.0, 8.0, length=grid_size)
    ys = range(-8.0, 8.0, length=grid_size)

    function sweep(mesh)
        po_grid = map(product(ys, xs)) do (y, x)
            heading = atan(drone_pos[2] - y, drone_pos[1] - x)
            actor = ActorState(x, y, actor_z, heading, mesh, 1)
            po, _ = eval_ppa_static(actor, drone_vec)
            return po
        end
        ps_grid = map(product(ys, xs)) do (y, x)
            heading = atan(drone_pos[2] - y, drone_pos[1] - x)
            actor = ActorState(x, y, actor_z, heading, mesh, 1)
            _, ps = eval_ppa_static(actor, drone_vec)
            return ps
        end
        return po_grid, ps_grid
    end

    po_quad, ps_quad = sweep(mesh_quad)
    po_tri,  ps_tri  = sweep(mesh_tri)

    max_val = global_max === nothing ?
        maximum([po_quad..., ps_quad..., po_tri..., ps_tri...]) : global_max

    function make_panel(grid, title)
        p = heatmap(xs, ys, grid,
            c=:jet, clims=(0.0, max_val),
            title=title,
            xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
            xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
        )
        scatter!(p, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
        quiver!(p, [drone_pos[1]], [drone_pos[2]],
            quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
        return p
    end

    p_orig_quad = make_panel(po_quad, "Original PPA — Quad Mesh")
    p_smooth_quad = make_panel(ps_quad, "Smooth PPA — Quad Mesh")
    p_orig_tri = make_panel(po_tri, "Original PPA — Triangulated Mesh")
    p_smooth_tri = make_panel(ps_tri, "Smooth PPA — Triangulated Mesh")

    p_combined = plot(p_orig_quad, p_smooth_quad, p_orig_tri, p_smooth_tri,
        layout=(2, 2), size=(1400, 1240),
        plot_title="PPA: Quad Faces vs. Triangulated Faces (Actor Heading Toward Drone)")

    out_file = joinpath(@__DIR__, "drone_experiments", "PPA_Static_QuadVsTriangulated_Grid.png")
    savefig(p_combined, out_file)
    println("\n✓ Saved grid comparison image → $out_file")

    return p_combined
end

run_static_comparison_quad_vs_tri()
