#Contains code for the triangulated (OBJ-loaded) actor mesh: geometry only, no
#visualization dependencies, so it can be used from Plots-based analysis scripts
#without pulling in GLMakie.

module ActorMeshTriangulated

using LinearAlgebra
using FileIO
using MeshIO
using GeometryBasics

export TriFace, TriMeshStruct, build_tri_mesh
export actor_world_vertices, actor_world_face_center, actor_world_normal, face_dynamic_weight

struct TriFace
    index::Int
    name::Symbol            # body part material name (e.g. :Head_face)
    a::Float64              # angular decay rate for this part
    normal::Vector{Float64} # local-frame outward normal
    center::Vector{Float64} # local-frame centroid
    area::Float64
    corner_indices::Vector{Int}
end

struct TriMeshStruct
    vertices::Vector{Vector{Float64}}   # all local-frame vertices
    faces::Vector{TriFace}
    height::Float64                     # z-extent, used to approximate actor center height
end

function triangle_geometry(m)
    coords = GeometryBasics.coordinates(m)
    tris = GeometryBasics.faces(m)
    normals = Vector{Vector{Float64}}(undef, length(tris))
    centers = Vector{Vector{Float64}}(undef, length(tris))
    areas = Vector{Float64}(undef, length(tris))
    for (i, tri) in enumerate(tris)
        v1 = Vector{Float64}(coords[tri[1]])
        v2 = Vector{Float64}(coords[tri[2]])
        v3 = Vector{Float64}(coords[tri[3]])
        n = cross(v2 .- v1, v3 .- v1)
        areas[i] = norm(n) / 2
        normals[i] = areas[i] > 0 ? n ./ norm(n) : n
        centers[i] = (v1 .+ v2 .+ v3) ./ 3
    end
    return normals, centers, areas, tris
end

function obj_material_face_groups(path::String)
    groups = Dict{Symbol, Vector{Int}}()
    current = :default
    face_idx = 0
    for line in eachline(path)
        line = strip(line)
        if startswith(line, "usemtl")
            current = Symbol(split(line)[2])
        elseif startswith(line, "f ")
            face_idx += 1
            push!(get!(groups, current, Int[]), face_idx)
        end
    end
    return groups
end

# Loads the triangulated actor mesh from `obj_path` and builds a `TriMeshStruct`.
# Skips the untagged `default` material group and any bottom-facing triangle
# (normal dominated by -z), since neither is ever camera-visible.
# `part_decay` maps each body-part material name (e.g. `:Head_face`) to its
# angular decay rate `a`, used by `face_dynamic_weight`.
function build_tri_mesh(obj_path::String; part_decay::Dict{Symbol,Float64})
    m = load(obj_path)
    coords = GeometryBasics.coordinates(m)
    vertices = [Vector{Float64}(v) for v in coords]

    groups = obj_material_face_groups(obj_path)
    normals, centers, areas, tris = triangle_geometry(m)

    faces = TriFace[]
    for (part, idxs) in groups
        part == :default && continue
        haskey(part_decay, part) || error("No decay rate `a` provided for part $part")
        a = part_decay[part]
        for i in idxs
            n = normals[i]
            argmax(abs.(n)) == 3 && n[3] < 0 && continue   # skip bottom-facing faces
            corner_indices = [convert(Int, tris[i][1]), convert(Int, tris[i][2]), convert(Int, tris[i][3])]
            push!(faces, TriFace(i, part, a, n, centers[i], areas[i], corner_indices))
        end
    end

    zs = [v[3] for v in vertices]
    height = maximum(zs) - minimum(zs)

    return TriMeshStruct(vertices, faces, height)
end

# ── World-frame transforms (2D yaw about z, matching ActorMesh.jl's convention) ──

function actor_world_vertices(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64)
    c, s = cos(heading), sin(heading)
    world_vertices = Vector{Vector{Float64}}(undef, length(mesh.vertices))
    for (i, v) in enumerate(mesh.vertices)
        wx = x + c*v[1] - s*v[2]
        wy = y + s*v[1] + c*v[2]
        wz = z + v[3]
        world_vertices[i] = [wx, wy, wz]
    end
    return world_vertices
end

function actor_world_face_center(mesh::TriMeshStruct, face::TriFace, x::Float64, y::Float64, z::Float64, heading::Float64)
    c, s = cos(heading), sin(heading)
    wx = x + face.center[1]*c - face.center[2]*s
    wy = y + face.center[1]*s + face.center[2]*c
    wz = z + face.center[3]
    return [wx, wy, wz]
end

function actor_world_normal(face::TriFace, heading::Float64)
    c, s = cos(heading), sin(heading)
    nx = face.normal[1]*c - face.normal[2]*s
    ny = face.normal[1]*s + face.normal[2]*c
    nz = face.normal[3]
    return [nx, ny, nz]
end

# ── Angular preferred-view weight: exp(-a * |θ_face - θ_pref|) ──

face_local_azimuth(n_local::Vector{Float64}) = atan(n_local[2], n_local[1])

function angular_difference(θ1::Float64, θ2::Float64)
    Δ = θ1 - θ2
    return abs(atan(sin(Δ), cos(Δ)))
end

# weight = exp(-a * |θ_face - θ_pref|), where θ_face is this face's normal
# azimuth rotated into world frame by `heading`, θ_pref is the bearing from
# the actor to the camera, and `a = face.a` is this face's part-specific decay
# rate. Weight is 1.0 when the face points exactly at the camera, decaying
# exponentially as it turns away.
#
# NOTE: this mesh is a symmetric hexagonal prism — every body part's triangles
# repeat at the same six azimuths (60° apart) with no geometric "front", so
# comparing each triangle's own normal to θ_pref is meaningless (there is
# always some triangle within 30° of facing the camera, for any heading).
# Instead compare the actor's overall `heading` to θ_pref, so the weight
# reflects which way the actor as a whole is facing.
function face_dynamic_weight(face::TriFace, heading::Float64, actor_pos::Vector{Float64}, camera_pos::Vector{Float64})
    θ_pref = atan(camera_pos[2] - actor_pos[2], camera_pos[1] - actor_pos[1])
    return exp(-face.a * angular_difference(heading, θ_pref))
end

end # module
