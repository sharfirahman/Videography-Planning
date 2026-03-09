module ActorTrajectory
#using ..ActorMesh:ActorMeshStruct, build_actor_mesh,actor_world_vertices, actor_world_face_center, actor_world_normal
using ..ActorMesh
using LinearAlgebra
using Plots
gr()


export ActorState, circular_trajectory, figure_eight_trajectory


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


end
