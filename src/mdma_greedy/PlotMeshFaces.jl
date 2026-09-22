# PlotMeshFaces.jl
# Renders the human/cylinder mesh using its ORIGINAL face structure straight
# from the .obj file (no MeshIO triangulation) — colors each face by its
# corner count so you can see exactly which faces are quads vs. triangles.

ENV["GKSwstype"] = "100"

using GLMakie, GLMakie.FileIO, GeometryBasics

const OBJ_PATH = joinpath(@__DIR__, "simple_human_rotated_with_texture.obj")
const OUT_PATH = joinpath(@__DIR__, "drone_experiments", "MeshFaces_TriVsQuad.png")

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

mkpath(joinpath(@__DIR__, "drone_experiments"))

vertices, face_index_lists = parse_obj_quads(OBJ_PATH)
println("Loaded $(length(vertices)) vertices, $(length(face_index_lists)) faces")

fig = Figure(size=(1000, 1000))
ax = Axis3(fig[1, 1], aspect=:data, title="Original mesh faces (orange = triangle, steelblue = quad)")
hidedecorations!(ax)
hidespines!(ax)

for idxs in face_index_lists
    corners = [Point3f(vertices[i]...) for i in idxs]
    n = length(corners)
    color = n == 3 ? :orange : :steelblue

    # Fan-triangulate purely for rendering — the faces are planar, so this
    # doesn't change the shape, just lets GLMakie draw the flat polygon.
    tri_faces = [GLTriangleFace(1, j, j + 1) for j in 2:(n - 1)]
    gmesh = GeometryBasics.Mesh(corners, tri_faces)
    mesh!(ax, gmesh, color=color, shading=NoShading)

    # Outline each original face so the true face boundaries are visible.
    lines!(ax, vcat(corners, [corners[1]]), color=:black, linewidth=1.5)
end

save(OUT_PATH, fig)
println("Saved -> $OUT_PATH")
