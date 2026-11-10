using GeometryBasics
using Plots
# plotlyjs()

#rect = Rect(Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))

c =  Cylinder(Point3f(0, 0, 0), Point3f(0, 0, 1), 0.3f0)




c_faces = decompose(TriangleFace{Int}, c)

c_positions = decompose(Point3f, c)

mesh = Mesh(c_positions, c_faces)

#mesh = GeometryBasics.mesh(rect)

mesh_faces = faces(mesh)
println("Number of faces: ", length(mesh_faces))

xs = [p[1] for p in c_positions]
ys = [p[2] for p in c_positions]
zs = [p[3] for p in c_positions]

plt = plot(legend=false, title="Cylinder Mesh ($(length(mesh_faces)) faces)", camera=(45, 30))
for f in mesh_faces
    tri_xs = [xs[f[1]], xs[f[2]], xs[f[3]], xs[f[1]]]
    tri_ys = [ys[f[1]], ys[f[2]], ys[f[3]], ys[f[1]]]
    tri_zs = [zs[f[1]], zs[f[2]], zs[f[3]], zs[f[1]]]
    plot!(plt, tri_xs, tri_ys, tri_zs, linecolor=:black)
end
scatter!(plt, xs, ys, zs, color=:red, markersize=3)
savefig(plt, "cylinder_mesh_plot.png")
