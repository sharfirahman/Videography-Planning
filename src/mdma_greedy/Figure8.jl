using Pkg
Pkg.activate(".")



using ActorTrajectory
using DroneVisualization

println("Creating figure-8 trajectory...")

# Create figure-8 actor trajectory
actor_traj = figure_eight_trajectory(
    num_steps=250,
    scale=3.0,  # Size of the figure-8
    angular_velocity=0.5,  # Speed (completes one full figure-8 in ~25 seconds)
    faces=default_actor_faces(actor_height=0.8),
    actor_id=1
)

println("Created actor trajectory: $(length(actor_traj)) steps")

# Create drone trajectory (follows actor with offset)
drone_traj = Vector{Vector{Float64}}()
drone_offset = 2.5  # Drone stays 2.5m behind actor

for i in 1:length(actor_traj)
    actor = actor_traj[i]
    
    # Drone position: offset behind actor based on heading
    drone_heading = actor.heading
    drone_x = actor.x - drone_offset * cos(drone_heading)
    drone_y = actor.y - drone_offset * sin(drone_heading)
    drone_z = actor.z + 2.0  # 2m above actor
    
    # Drone state: [x,y,z,vx,vy,vz,theta,omega]
    vx = 0.3 * cos(drone_heading)  # Forward velocity
    vy = 0.3 * sin(drone_heading)
    vz = 0.0
    omega = 0.05  # Small yaw rate
    
    push!(drone_traj, [drone_x, drone_y, drone_z, vx, vy, vz, drone_heading, omega])
end

println("Created drone trajectory: $(length(drone_traj)) steps")

# Generate the visualization!
println("\nGenerating animation...")
anim = animate_drone_and_actor(
    actor_traj, 
    drone_traj;
    anim_file="drone_figure8_tracking.gif",
    fps=15,
    xlims=(-5, 5),
    ylims=(-4, 4), 
    zlims=(0, 5),
    camera=(30, 45)
)

