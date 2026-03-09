include("./src/mdma_greedy/ActorMesh.jl")
include("./src/mdma_greedy/ActorTrajectory.jl")

using Main.ActorMesh
using Main.ActorTrajectory

# Use same settings from FPV configuration
mesh = build_actor_mesh()
actor_traj = figure_eight_trajectory(
    mesh;
    num_steps        = 150,
    scale            = 3.0,
    angular_velocity = 2π / 15.0,
    dt               = 0.2, 
    actor_id         = 1
)

drone_traj = Vector{Vector{Float64}}()

DRONE_BEHIND  = 2.0 
DRONE_ABOVE   = 1.5 

for actor in actor_traj
    heading = actor.heading
    drone_x = actor.x - DRONE_BEHIND * cos(heading)
    drone_y = actor.y - DRONE_BEHIND * sin(heading)
    drone_z = actor.z + DRONE_ABOVE
    vx      = 0.3 * cos(heading)
    vy      = 0.3 * sin(heading)
    # state: x, y, z, vx, vy, vz, heading, omega
    push!(drone_traj, [drone_x, drone_y, drone_z, vx, vy, 0.0, heading, 0.1])
end

# Save to CSV
filename_out = "figure8_drone_trajectory.csv"
open(filename_out, "w") do io
    write(io, "step,x,y,z,vx,vy,vz,heading,omega\n")
    for (i, d) in enumerate(drone_traj)
        write(io, "$i, $(d[1]), $(d[2]), $(d[3]), $(d[4]), $(d[5]), $(d[6]), $(d[7]), $(d[8])\n")
    end
end

println("Successfully saved $(length(drone_traj)) data points to $filename_out")
