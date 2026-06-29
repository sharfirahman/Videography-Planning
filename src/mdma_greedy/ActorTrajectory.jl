#Contains functions to generate different trajectories for the actor.


module ActorTrajectory
#using ..ActorMesh:ActorMeshStruct, build_actor_mesh,actor_world_vertices, actor_world_face_center, actor_world_normal
using ..ActorMesh
using LinearAlgebra
using Plots
gr()


export ActorState, circular_trajectory, figure_eight_trajectory, lissajous_trajectory, crossing_splines_trajectory


struct ActorState
    x::Float64
    y::Float64
    z::Float64
    heading::Float64
    mesh::ActorMeshStruct
    actor_id::Int
end


#Want the actor to go in straight line to see how the drone acts
function general_trajectory(
    mesh;
    num_steps::Int =100,
    init_position::Vector{Float64} = [0.0,0.0,0.0],
    heading::Float64 = 0.0,
    velocity::Float64 = 0.1,
    dt::Float64 = 0.2,
    actor_id::Int =1
)

    trajectory = ActorState[]

    for t in 1:num_steps
        x = init_position[1]+velocity *dt *t *cos(heading)
        y = init_position[2]+velocity *dt *t *sin(heading)
        z = init_position[3]

        actor = ActorState(x,y,z,heading,mesh,actor_id)
        push!(trajectory,actor)
    end

    return trajectory

end

#Wanna test the circular trajectory too as implemented in MPC
function circular_trajectory(
    mesh;
    num_steps::Int =200,
    init_position::Vector{Float64} = [0.0,0.0,0.0],
    radius::Float64 = 3.0,
    start_angle::Float64 = 0.0,
    angular_velocity::Float64 = 0.6,
    dt::Float64 = 0.2,
    actor_id::Int =1
)

    trajectory = ActorState[]

    for t in 1:num_steps

        angle = start_angle + angular_velocity *t *dt 
        x = init_position[1]+radius *cos(angle)
        y = init_position[2]+radius *sin(angle)
        z = init_position[3]

        heading =angle + pi/2


        actor = ActorState(x,y,z,heading,mesh,actor_id)
        push!(trajectory,actor)
    end

    return trajectory

end

function figure_eight_trajectory(
    mesh;
    num_steps::Int = 200,
    init_position::Vector{Float64} = [0.0, 0.0, 0.0],
    scale::Float64 = 3.0,  # Size of the figure-8
    angular_velocity::Float64 = 0.4,  # Speed around the curve
    dt::Float64 = 0.2,
    actor_id::Int = 1
)
    trajectory = ActorState[]
    
    for t in 1:num_steps
        # Parametric time variable
        θ = angular_velocity * t * dt
        

        x = init_position[1] + scale * cos(θ)
        y = init_position[2] + scale * sin(θ) * cos(θ)
        z = init_position[3]
        #heading
        dx = -scale * sin(θ)
        dy = scale * cos(2 * θ)
        
        heading = atan(dy, dx)
        
        actor = ActorState(x, y, z, heading, mesh, actor_id)
        push!(trajectory, actor)
    end
    
    return trajectory
end

# Lissajous (cross-mix) trajectory
# Creates a crossing/weaving pattern by using different frequencies
# for x and y. E.g. omega_x != omega_y gives looping crossing figures.
function lissajous_trajectory(
    mesh;
    num_steps::Int                 = 200,
    init_position::Vector{Float64}  = [0.0, 0.0, 0.0],
    amplitude_x::Float64           = 4.5,        # half-width in x
    amplitude_y::Float64           = 4.0,        # half-height in y
    omega_x::Float64               = 2π / 15.0,  # x angular frequency
    omega_y::Float64               = 2π / 10.0,  # y angular frequency (different → crossings)
    phase::Float64                 = π / 2,      # phase offset creates the crossing
    dt::Float64                    = 0.2,
    actor_id::Int                  = 1
)
    trajectory = ActorState[]
    for t in 1:num_steps
        τ = t * dt
        x  = init_position[1] + amplitude_x * sin(omega_x * τ + phase)
        y  = init_position[2] + amplitude_y * sin(omega_y * τ)
        z  = init_position[3]
        # heading = tangent direction of the parametric curve
        dx = amplitude_x * omega_x * cos(omega_x * τ + phase)
        dy = amplitude_y * omega_y * cos(omega_y * τ)
        heading = atan(dy, dx)
        push!(trajectory, ActorState(x, y, z, heading, mesh, actor_id))
    end
    return trajectory
end

# Crossing Splines Trajectory
# Simulates the pattern from the user image: paths start from opposite corners,
# curve inwards towards the center, cross each other, and flare out.
function crossing_splines_trajectory(
    mesh;
    num_steps::Int                 = 200,
    start_pos::Vector{Float64}     = [-5.0,  5.0, 0.0],  # Top-left or bottom-left
    end_pos::Vector{Float64}       = [ 5.0, -5.0, 0.0],  # Bottom-right or top-right
    curvature::Float64             = 3.0,                # How radically it bends in the middle
    dt::Float64                    = 0.2,
    actor_id::Int                  = 1
)
    trajectory = ActorState[]
    
    # We'll use a smooth step (cubic Hermite interpolation) to generate S-curves
    # that meet in the middle and cross over.
    for t in 0:(num_steps-1)
        # Normalized progress [0, 1]
        s = t / (num_steps - 1)
        
        # Smoothstep: 3s^2 - 2s^3 (slow start, fast middle, slow end)
        smooth_s = s * s * (3.0 - 2.0 * s)
        
        # Linear interpolation for the primary direction of travel (X usually)
        x = start_pos[1] + (end_pos[1] - start_pos[1]) * s
        
        # Base straight line for Y
        y_linear = start_pos[2] + (end_pos[2] - start_pos[2]) * s
        
        # Add an S-curve deviation to Y to make them 'flare' into the crossing
        # Sine wave half-period creates a bulge that pulls it toward/away from center
        y_deviation = curvature * sin(π * s)
        
        # If the path goes top-to-bottom, we bend one way. Bottom-to-top bends the other.
        direction_sign = sign(start_pos[2])
        y = y_linear - direction_sign * y_deviation
        
        z = start_pos[3]

        # Calculate heading numerically using a small delta
        ds = 0.01
        s_next = min(1.0, s + ds)
        smooth_s_next = s_next * s_next * (3.0 - 2.0 * s_next)
        
        x_next = start_pos[1] + (end_pos[1] - start_pos[1]) * s_next
        y_linear_next = start_pos[2] + (end_pos[2] - start_pos[2]) * s_next
        y_dev_next = curvature * sin(π * s_next)
        y_next = y_linear_next - direction_sign * y_dev_next
        
        heading = atan(y_next - y, x_next - x)
        
        push!(trajectory, ActorState(x, y, z, heading, mesh, actor_id))
    end
    
    return trajectory
end

end
