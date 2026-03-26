# actor_visualization.jl
# Visualizes the 3D ActorMesh using CairoMakie (supports solid flat colors with shading=NoShading)

ENV["GKSwstype"] = "100"

include("ActorMesh.jl")
using .ActorMesh
using CairoMakie
using LinearAlgebra
const GLTriangleFace = Makie.GeometryBasics.GLTriangleFace
const Point3f        = Makie.GeometryBasics.Point3f

cam_dir(az, el) = [cos(el)*cos(az), cos(el)*sin(az), sin(el)]

function draw_actor!(ax, mesh, world_verts, az, el)
    cdir = cam_dir(az, el)

    # Back-face culling: only draw faces pointing toward camera
    visible_faces = filter(f -> dot(f.normal, cdir) > 0.0, mesh.faces)

    # Sort back-to-front (Painter's Algorithm)
    sort!(visible_faces, by = f -> dot(f.normal, cdir))

    for face in visible_faces
        pts  = [world_verts[idx] for idx in face.corner_indices]
        p1, p2, p3, p4 = pts[1], pts[2], pts[3], pts[4]
        verts = [Point3f(p...) for p in (p1, p2, p3, p4)]
        tris  = [GLTriangleFace(1,2,3), GLTriangleFace(1,3,4)]
        col   = parse(Makie.Colors.Colorant, string(face.color))
        mesh!(ax, verts, tris; color=col, shading=NoShading)
    end

    # Wireframe edges
    for (i1, i2) in mesh.edges
        lines!(ax, [Point3f(world_verts[i1]...), Point3f(world_verts[i2]...)];
               color=:black, linewidth=1.5)
    end
end

function make_frame!(fig, mesh, world_verts, az, el)
    empty!(fig)
    ax = Axis3(fig[1,1]; aspect=:data, elevation=el, azimuth=az,
               protrusions=0, viewmode=:fit)
    hidedecorations!(ax)
    hidespines!(ax)
    draw_actor!(ax, mesh, world_verts, az, el)
end

function plot_actor_mesh()
    println("Building ActorMesh...")
    mesh = build_actor_mesh(
        actor_width  = 0.3,
        actor_depth  = 0.3,
        actor_height = 1.0
    )
    world_verts = actor_world_vertices(mesh, 0.0, 0.0, 0.0, 0.0)

    el = π/6

    # Static image
    fig = Figure(size=(800, 600))
    make_frame!(fig, mesh, world_verts, π/4, el)
    println("Saving static image to actor_3d_static.png...")
    save("actor_3d_static.png", fig)

    # Rotating MP4 with per-frame back-face culling (MP4 preserves full color, GIF doesn't)
    println("Saving rotation animation to actor_3d_circle.mp4...")
    fig2 = Figure(size=(800, 600))
    record(fig2, "actor_3d_circle.mp4", range(0, 2π, length=72); framerate=15) do az
        make_frame!(fig2, mesh, world_verts, az, el)
    end

    println("✓ Done!")
end

plot_actor_mesh()