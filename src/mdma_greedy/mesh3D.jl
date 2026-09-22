#using CairoMakie, FileIO, GeometryBasics

using GLMakie, FileIO, GeometryBasics

m = load("/home/sharfi/dev/MultiDroneMultiActorFilming/src/mdma_greedy/simple_human_rotated_color.obj")

verts = GeometryBasics.coordinates(m)
faces = GeometryBasics.faces(m)


for i in 1:length(faces)
    println(faces[i])
end



# fig = mesh(m; color=:blue, shading = NoShading,  axis=(; show_axis=true))
fig = mesh(verts, faces; color=faces, shading=NoShading, axis=(; show_axis=true))

save("mesh3D.png", fig)