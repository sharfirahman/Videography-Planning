using GeometryBasics
using CairoMakie

rect = Rect(Vec(0,0,0), Vec(1,1,1))

square_faces =  decompose(SquareFace{Int}, rect)

mesh = GeometryBasics.mesh(rect)

println("Number of faces: ", length(faces(mesh)))

fig = Figure()
ax = Axis3(fig[1, 1], aspect=:data)
mesh!(ax, mesh, color=:lightblue, shading=Makie.automatic)
wireframe!(ax, mesh, color=:black, linewidth=1)
save("mesh_plot.png", fig)
display(fig)
