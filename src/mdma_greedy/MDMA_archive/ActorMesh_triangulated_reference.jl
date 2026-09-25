module ActorMesh_triangulated


using GLMakie, FileIO, GeometryBasics, LinearAlgebra
GLMakie.activate!()

obj_path = "/home/sharfi/dev/MultiDroneMultiActorFilming/src/mdma_greedy/simple_human_rotated_color.obj"
m = load(obj_path)

#Create struct for each triangleface to store its properties

struct TriFace
    index::Int
    name::Symbol
    weight::Float64
    normal::Vector{Float64}
    center::Vector{Float64}
    area::Float64
    corner_indices::Vector{Int}
end

#Calculate normal vector for each triangleface

function triangle_geometry(m)
    coords = GeometryBasics.coordinates(m)
    tris = GeometryBasics.faces(m)
    normals = Vector{Vector{Float64}}(undef, length(tris))
    centers = Vector{Vector{Float64}}(undef, length(tris))
    areas = Vector{Float64}(undef, length(tris))
    for (i, tri) in enumerate(tris)
        v1, v2, v3 = Vector{Float64}(coords[tri[1]]), Vector{Float64}(coords[tri[2]]), Vector{Float64}(coords[tri[3]])
        n = cross(v2 .- v1, v3 .- v1)
        areas[i] = norm(n) / 2
        normals[i] = areas[i] > 0 ? n ./ norm(n) : n
        centers[i] = (v1 .+ v2 .+ v3) ./ 3
    end
    return normals, centers, areas, tris
end

#Read obj material face groups - head, body, feet, top faces

function obj_material_face_groups(path::String)
    groups = Dict{Symbol,Vector{Int}}()
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

#Calculate face local azimuth

face_local_azimuth(n_local::Vector{Float64}) = atan(n_local[2], n_local[1])
#Calculate angular difference - between 0 and pi
function angular_difference(θ1::Float64, θ2::Float64)
    Δ = θ1 - θ2
    return abs(atan(sin(Δ), cos(Δ)))
end

#Calculate preferred view weight

function preferred_view_weight(
    n_local::Vector{Float64}, θ_pref::Float64, a::Float64
)
    θ_face = face_local_azimuth(n_local)
    #θ_pref = 0.0
    return exp(-a * angular_difference(θ_face, θ_pref))
end



function build_weighted_faces(m, obj_path::String,
    region_weight::Dict{Symbol,Float64})
    groups = obj_material_face_groups(obj_path)
    normals, centers, areas, tris = triangle_geometry(m)
    faces = TriFace[]
    for (part, idxs) in groups
        part == :default && continue
        rw = region_weight[part]

        for i in idxs
            n = normals[i]
            #argmax(abs.(n)) == 3 && n[3] < 0 && continue   # skip bottom-facing faces
            w = rw * preferred_view_weight(n, 0.0, angle_decay)
            push!(faces, TriFace(i, part, w, n, centers[i], areas[i], [convert(Int, tris[i][1]), convert(Int, tris[i][2]), convert(Int, tris[i][3])]))
        end
    end
    return faces
end

#--------------------------------------------------------------------------------------
#Putting it all together - call the function and visualize
angle_decay = 2 / pi
region_weight = Dict(:Head_face => 1.0, :Body_face => 0.75, :Feet_face => 0.5, :Top_face => 0.8)
faces = build_weighted_faces(m, obj_path, region_weight)

# ── Visualize mesh with every triangle's normal ("heading") drawn as an arrow ──
fig = Figure()
ax = Axis3(fig[1, 1], aspect=:data)
mesh!(ax, m)
GLMakie.wireframe!(ax, m, color=:black, linewidth=0.5)

normals, centers, areas, tris = triangle_geometry(m)
offset = 0.015
keep = [!(argmax(abs.(n)) == 3 && n[3] < 0) for n in normals]   # exclude bottom-facing
origins = [Point3f((c .+ offset .* n)...) for (c, n, k) in zip(centers, normals, keep) if k]
directions = [Vec3f(n...) for (n, k) in zip(normals, keep) if k]

arrows3d!(ax, origins, directions;
    color=:red, lengthscale=0.3,
    shaftradius=0.015, tipradius=0.035, tiplength=0.08, taillength=0)

display(fig)
save("src/mdma_greedy/drone_experiments/actor_face_headings.png", fig)
end