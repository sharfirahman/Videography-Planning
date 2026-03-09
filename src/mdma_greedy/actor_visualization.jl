
include("ActorTrajectory.jl")
include("DroneVisualization.jl")
using MPC
using .ActorTrajectory
using .DroneVisualization
using Plots
gr()

"""
draw_single_actor(actor; save_file="")

Draw a single actor with colored faces in 3D.

# Arguments
- `actor::ActorState`: The actor to visualize
- `save_file::String`: Optional filename to save the plot
"""
function draw_single_actor(actor::ActorState; save_file="actor_3d.png")
    # Get the 8 corners of the actor box
    corners = get_actor_corners(actor)
    
    # Create 3D plot
    p = plot(
        xlabel="X (m)", 
        ylabel="Y (m)", 
        zlabel="Z (m)",
        title="Actor with Colored Faces",
        legend=false,
        size=(800, 800),
        camera=(45, 30),  # View angle (azimuth, elevation)
        aspect_ratio=:equal
    )
    
    # Draw colored faces
    draw_colored_actor!(p, actor, corners)
    
    # Add coordinate axes at origin for reference
    plot!(p, [0, 1], [0, 0], [0, 0], color=:red, linewidth=2, label="X", arrow=true)
    plot!(p, [0, 0], [0, 1], [0, 0], color=:green, linewidth=2, label="Y", arrow=true)
    plot!(p, [0, 0], [0, 0], [0, 1], color=:blue, linewidth=2, label="Z", arrow=true)
    
    # Save if filename provided
    if !isempty(save_file)
        savefig(p, save_file)
        println("✓ Saved to: $save_file")
    end
    
    return p
end

"""
Main demo function
"""
function main()
    println("="^60)
    println("Visualizing Actor with Colored Faces")
    println("="^60)
    
    # Create a single actor at origin, facing East (heading=0)
    faces = default_actor_faces()
    actor = ActorState(0.0, 0.0, 0.5, 0.0, faces, 1)
    
    println("\nActor properties:")
    println("  Position: ($(actor.x), $(actor.y), $(actor.z))")
    println("  Heading: $(round(rad2deg(actor.heading), digits=1))°")
    println("\nFace colors:")
    for face in actor.faces
        println("  $(face.name): $(face.color)")
    end
    
    # Draw and save
    p = draw_single_actor(actor, save_file="actor_3d.png")
    
    # Display (if in interactive environment)
    display(p)
    
    println("\n✓ Done! Check actor_3d.png")
    println("="^60)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end