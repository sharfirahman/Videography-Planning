

# include("MDMA.jl")


# using .MDMA
ENV["GKSwstype"] = "100"



using .MPC
using ..ActorMesh
using ..ActorTrajectory
using ..DroneVisualizationFPV



function run_drone_actor_simulation(;
    num_steps::Int = 200,
    radius::Float64 = 3.0,
    follow_distance::Float64 = 2.5,
    #follow_angle::Float64 = 180.0,
    output_file::String = "drone_actor_tracking.gif"
)
    println("="^70)
    println("Drone-Actor Tracking Simulation")
    println("="^70)

    println("\n[1/3] Generating actor trajectory...")
    
    # actor_trajectory = circular_trajectory(
    #     num_steps = num_steps,
    #     radius = radius,
    #     angular_velocity = 0.0314
    # )

    actor_trajectory = circular_trajectory(
    build_actor_mesh();
    num_steps        = num_steps,
    radius           = radius,
    angular_velocity = 2π / 15.0,
    actor_id         = 1
    )
    
    println("  ✓ Generated $(length(actor_trajectory)) actor states")


    println("\n[2/3] Setting up MPC controller...")
    
    ax_max = 1.5
    az_max = 0.5 * 9.81
    alpha_max = π/2
    
    params = RobotParameters(
        10,                                          # Horizon
        0.2,                                         # Time step
        [-ax_max, -ax_max, -az_max, -alpha_max],   # Min control
        [ax_max, ax_max, az_max, alpha_max],        # Max control
        follow_distance,                             # Follow distance
        1.0                                 # Follow angle
    )
    
    println("  • Horizon: $(params.N) steps")
    println("  • Follow distance: $(follow_distance)m")

    println("\n Running MPC simulation...")
    
    initial_position = [5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    current_position = copy(initial_position)
    drone_trajectory = [copy(current_position)]
    
    for step in 1:(length(actor_trajectory) - 1)
        if step % 20 == 0
            print("  • Step $step/$(length(actor_trajectory)-1)\r")
        end
        
        horizon_end = min(step + params.N - 1, length(actor_trajectory))
        actor_horizon = actor_trajectory[step:horizon_end]
        
        pos_opt, vel_opt = SafeTrajectory(
            current_position,
            [0.0, 0.0, 0.0, 0.0],
            params,
            actor_horizon
        )
        
        current_control = vel_opt[:, 1]
        current_position = RobotDynamics(current_position, current_control, params.Ts)
        
        push!(drone_trajectory, copy(current_position))
    end
    

    
    animate_drone_and_actor(
        actor_trajectory,
        drone_trajectory,
        anim_file = output_file,
        fps = 12
    )
    
    println("Animation saved: $output_file")
    
    println("\n" * "="^70)
    println("✓ Done!")
    println("="^70)
    
    return actor_trajectory, drone_trajectory
end

# Run if executed directly
# if abspath(PROGRAM_FILE) == @__FILE__
#     run_drone_actor_simulation()
# end

run_drone_actor_simulation(num_steps= 200,
    radius= 3.0,
    follow_distance= 2.5,
    #follow_angle= 180.0,
    output_file= "drone_actor_tracking.gif")