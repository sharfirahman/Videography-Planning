

using ActorTrajectory 
using DroneVisualization

# Create circling actor trajectory 
actor_traj = circular_trajectory(
    num_steps=150,
    radius=3.0,
    angular_velocity=2π/15.0,  
    faces=default_actor_faces(actor_height=0.8),
    actor_id=1
)


drone_traj = Vector{Vector{Float64}}()
drone_offset = 2.0  # Drone stays 2m behind actor

for i in 1:length(actor_traj)
    actor = actor_traj[i]
    
    # Drone position: same heading, offset behind actor
    drone_heading = actor.heading
    drone_x = actor.x - drone_offset * cos(drone_heading)
    drone_y = actor.y - drone_offset * sin(drone_heading)
    drone_z = actor.z + 1.5  # 1.5m above actor
    
    # Drone state: [x,y,z,vx,vy,vz,theta,omega]
    vx = 0.3 * cos(drone_heading)  # Forward velocity
    vy = 0.3 * sin(drone_heading)
    vz = 0.0
    omega = 0.1  # Small yaw rate
    
    push!(drone_traj, [drone_x, drone_y, drone_z, vx, vy, vz, drone_heading, omega])
end

println("Created trajectories:")
println("   Actor: $(length(actor_traj)) steps")
println("   Drone: $(length(drone_traj)) steps")


anim = animate_drone_and_actor(
    actor_traj, 
    drone_traj;
    anim_file="drone_follows_actor_3d.gif",
    fps=12,
    xlims=(-6, 6),
    ylims=(-6, 6), 
    zlims=(0, 5),
    camera=(30, 45)
)

println("Animation saved: drone_follows_actor_3d.gif")
