using GeometryBasics
using Plots
# plotlyjs()

rect = Rect(Vec(0.0, 0.0, 0.0), Vec(1.0, 1.0, 1.0))

# Hexagonal prism: 6 named side faces + 1 top face = 7 faces total (no bottom)
radius = 0.3f0
height = 1.0f0

face_names = [:front, :front_left, :back_left, :back, :back_right, :front_right]
# Rim vertices sit at the midpoints between named directions so each side panel
# is angularly centered on its name (front centered at 0°, etc.)
rim_angles = [Float32(deg2rad(-30 + 60 * (i - 1))) for i in 1:6]

bottom_ring = [Point3f(radius * cos(a), radius * sin(a), 0.0f0) for a in rim_angles]
top_ring    = [Point3f(radius * cos(a), radius * sin(a), height) for a in rim_angles]
top_center  = Point3f(0.0f0, 0.0f0, height)

c_positions = vcat(bottom_ring, top_ring, [top_center])
# indices: bottom_ring -> 1:6, top_ring -> 7:12, top_center -> 13

side_faces = [QuadFace(i, mod1(i + 1, 6), mod1(i + 1, 6) + 6, i + 6) for i in 1:6]
top_face   = NgonFace{6,Int}(7, 8, 9, 10, 11, 12)

mesh_faces = vcat(side_faces, [top_face])
face_labels = vcat(face_names, [:top])

println("Number of faces: ", length(mesh_faces))
for (nm, f) in zip(face_labels, mesh_faces)
    println("  ", nm, " -> ", f)
end

xs = [p[1] for p in c_positions]
ys = [p[2] for p in c_positions]
zs = [p[3] for p in c_positions]

plt = plot(legend=false, title="Hexagonal Prism ($(length(mesh_faces)) faces)", camera=(45, 30))
for f in mesh_faces
    idx = vcat(collect(f), f[1])
    plot!(plt, xs[idx], ys[idx], zs[idx], linecolor=:black)
end
scatter!(plt, xs, ys, zs, color=:red, markersize=3)
savefig(plt, "hex_prism_plot.png")

