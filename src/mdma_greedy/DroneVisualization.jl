module DroneVisualization

using Plots
using LinearAlgebra

export draw_quadcopter, animate_drone_and_actor
#Geometry Helpers for drawing the actor and drones

function get_actor_corners(actor)
    c = cos(actor.heading)
    s = sin(actor.heading)
    
    # Get dimensions from face offsets
    width = actor.mesh.faces[3].offset[2] * 2   # left/right
    depth = actor.mesh.faces[1].offset[1] * 2   # front/back
    height = actor.mesh.faces[5].offset[3] * 2  # top/bottom
    
    # 8 corners in local frame
    corners_local = [
        [-depth/2, -width/2, -height/2],  # 1
        [depth/2, -width/2, -height/2],   # 2
        [depth/2, width/2, -height/2],    # 3
        [-depth/2, width/2, -height/2],   # 4
        [-depth/2, -width/2, height/2],   # 5
        [depth/2, -width/2, height/2],    # 6
        [depth/2, width/2, height/2],     # 7
        [-depth/2, width/2, height/2]     # 8
    ]
    
    # Transform to world frame
    corners_world = []
    for corner in corners_local
        x = actor.x + c * corner[1] - s * corner[2]
        y = actor.y + s * corner[1] + c * corner[2]
        z = actor.z + corner[3]
        push!(corners_world, [x, y, z])
    end
    
    return corners_world
end

function draw_quadcopter!(p, drone_state, arm_length=0.3, prop_radius=0.15)
    x, y, z, vx, vy, vz, theta, omega = drone_state
    
    # Body center
    scatter!(p, [x], [y], [z], 
            markersize=8, color=:darkred,
            markerstrokewidth=2, markerstrokecolor=:black, 
            label="")
    
    # Arms at 45 degree angles
    arm_angles = [pi/4, 3*pi/4, 5*pi/4, 7*pi/4]
    
    for (i, arm_offset) in enumerate(arm_angles)
        angle = theta + arm_offset
        
        # Arm endpoint
        arm_x = x + arm_length * cos(angle)
        arm_y = y + arm_length * sin(angle)
        arm_z = z
        
        # Draw arm
        plot!(p, [x, arm_x], [y, arm_y], [arm_z, arm_z],
              color=:black, linewidth=3, label="")
        
        # Motor
        scatter!(p, [arm_x], [arm_y], [arm_z], 
                markersize=4, color=:gray,
                markerstrokewidth=1, markerstrokecolor=:black,
                label="")
        
        # Propeller (circle)
        prop_theta = range(0, 2*pi, length=20)
        prop_x = arm_x .+ prop_radius * cos.(prop_theta) .* cos(angle) .- 
                         prop_radius * sin.(prop_theta) .* sin(angle)
        prop_y = arm_y .+ prop_radius * cos.(prop_theta) .* sin(angle) .+ 
                         prop_radius * sin.(prop_theta) .* cos(angle)
        prop_z = fill(arm_z + 0.05, length(prop_theta))
        
        # Alternate propeller colors
        prop_color = (i % 2 == 0) ? :red : :blue
        plot!(p, prop_x, prop_y, prop_z, 
              color=prop_color, linewidth=2, alpha=0.6, label="")
    end
    
    # Camera gimbal (points forward)
    camera_offset = 0.15
    cam_x = x + camera_offset * cos(theta)
    cam_y = y + camera_offset * sin(theta)
    cam_z = z - 0.1
    
    plot!(p, [x, cam_x], [y, cam_y], [z, cam_z], 
          color=:lime, linewidth=3, label="")
    scatter!(p, [cam_x], [cam_y], [cam_z], 
            markersize=3, color=:lime, marker=:square, label="")
end

function draw_colored_actor!(p, actor, corners)
    # Define which corners form each face
    face_indices = Dict(
        :front => [2, 3, 7, 6],    # +X face
        :back => [1, 4, 8, 5],     # -X face
        :left => [1, 2, 6, 5],     # +Y face
        :right => [4, 3, 7, 8],    # -Y face
        :top => [5, 6, 7, 8],      # +Z face
        :bottom => [1, 2, 3, 4]    # -Z face
    )

    for (face_name, indices) in face_indices
        face_corners = [corners[idx] for idx in indices]
        
        # Find color for this face
        face_color = :orange  # default
        for face in actor.mesh.faces
            if face.name == face_name
                face_color = face.color
                break
            end
        end
        
        # Extract coordinates and close polygon
        xs_face = [c[1] for c in face_corners]
        ys_face = [c[2] for c in face_corners]
        zs_face = [c[3] for c in face_corners]
        
        push!(xs_face, xs_face[1])
        push!(ys_face, ys_face[1])
        push!(zs_face, zs_face[1])
        
        # Draw filled face
        plot!(p, xs_face, ys_face, zs_face,
              fillrange=0, fillalpha=0.7, fillcolor=face_color,
              linewidth=2, linecolor=:black, label="")
    end
    
    # Actor heading arrow
    arrow_len = 0.8
    hx = actor.x + arrow_len * cos(actor.heading)
    hy = actor.y + arrow_len * sin(actor.heading)
    
    quiver!(p, [actor.x], [actor.y], [actor.z],
            quiver=([hx - actor.x], [hy - actor.y], [0]),
            color=:green, linewidth=4, arrow=true, label="")
end

function animate_drone_and_actor(
    actor_trajectory::Vector,
    drone_trajectory::Vector{Vector{Float64}};
    anim_file::String = "drone_actor_tracking.gif",
    fps::Int = 10,
    xlims::Tuple = (-4, 4),
    ylims::Tuple = (-4, 4),
    zlims::Tuple = (0, 4),
    camera::Tuple = (25, 45)
    )
    # Extract positions
    actor_xs = [a.x for a in actor_trajectory]
    actor_ys = [a.y for a in actor_trajectory]
    actor_zs = [a.z for a in actor_trajectory]
    
    drone_xs = [d[1] for d in drone_trajectory]
    drone_ys = [d[2] for d in drone_trajectory]
    drone_zs = [d[3] for d in drone_trajectory]
    
    num_frames = min(length(actor_trajectory), length(drone_trajectory))
    
    
    
    anim = @animate for i in 1:num_frames
        actor = actor_trajectory[i]
        drone = drone_trajectory[i]
        
        # Get actor geometry
        corners = get_actor_corners(actor)
        
        # Create plot
        p = plot(
            xlabel="X (m)", 
            ylabel="Y (m)", 
            zlabel="Z (m)",
            title="Drone Following Actor - Frame $i/$num_frames",
            legend=:topright,
            size=(1000, 800),
            camera=camera,
            xlims=xlims,
            ylims=ylims,
            zlims=zlims
        )
        
        # Actor path (blue)
        plot!(p, actor_xs[1:i], actor_ys[1:i], actor_zs[1:i],
              linewidth=3, color=:blue, alpha=0.6, label="Actor Path")
        
        # Drone path (red)
        plot!(p, drone_xs[1:i], drone_ys[1:i], drone_zs[1:i],
              linewidth=3, color=:red, alpha=0.6, label="Drone Path")
        
        # Draw colored actor
        draw_colored_actor!(p, actor, corners)
        
        # Draw quadcopter
        draw_quadcopter!(p, drone, 0.3, 0.15)
        
        # Connection line (drone to actor)
        plot!(p, [drone[1], actor.x], [drone[2], actor.y], [drone[3], actor.z],
              linestyle=:dash, color=:gray, linewidth=1, alpha=0.4, label="")
        
        # Start markers
        scatter!(p, [actor_xs[1]], [actor_ys[1]], [actor_zs[1]],
                markersize=6, color=:lightblue, marker=:square, label="Start")
        scatter!(p, [drone_xs[1]], [drone_ys[1]], [drone_zs[1]],
                markersize=6, color=:pink, marker=:square, label="")
        
        # if i % 40 == 0
        #     println("   Frame $i/$num_frames rendered")
        # end
    end
    
    # Save animation
    println("Saving animation to $anim_file...")
    gif(anim, anim_file, fps=fps)

    
    return anim
end

end