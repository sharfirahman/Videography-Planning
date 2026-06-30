module MPC


const g = 9.81


using LinearAlgebra
using Profile
using Ipopt
using JuMP 
using Plots
using ..ActorMesh
using ..ActorTrajectory





export RobotParameters, RobotDynamics, SafeTrajectory, SafeTrajectoryMultiActor, PlanTrajectory
export pinhole_ppa_reward

struct RobotParameters
    N::Int          #Finite Time Horizon
    Ts:: Float64    #Time step
    u_min::Vector{Float64}  # input lower bounds
    u_max::Vector{Float64}  # input upper bounds
    target_distance::Float64   #Desired filming distance from the actor
    safety_dist::Float64       #Minimum safe distance from the actor (< target_distance)
    #follow_angle::Float64 #Added this angle relative to actor's heading
 end

# Double integrator model


# """
# state vector: x,y,z,x_velocity,y_velocity,z_velocity, theta(yaw_angle)
# control vector: ax,ay,az,atheta (all the accelerations as a control input) 
# """

function RobotDynamics(state,control,dt)
 
    x,y,z,vx,vy,vz, theta,omega = state
    ax,ay,az,atheta = control # keeping linear accelaration and angular accelaration as control input

    #Translating Body accelaration in the world frame

    ax_w = ax * cos(theta) - ay * sin(theta)
    ay_w = ax * sin(theta) + ay * cos(theta)


    #velocities

    vx_new = vx + ax_w*dt  #v = v_0 + at
    vy_new = vy + ay_w*dt
    vz_new = vz + az*dt 

    #new positions for the quadrotor

    x_new = x + vx*dt
    y_new = y + vy*dt
    z_new = z + vz*dt

    #angles
    omega_new = omega + atheta * dt
    theta_new = theta + omega*dt                        #yaw_angle
    theta_new = atan(sin(theta_new), cos(theta_new))  #normalization

    return [x_new, y_new, z_new, vx_new, vy_new, vz_new, theta_new,omega_new ]

end

#Add the reward function helpers
# Smooth approximation of max(0, -dot(distance, face_normal))
function isvisible(
    distance::Vector,
    face_normal::Vector
)
    x = -dot(distance, face_normal)
    return (x + sqrt(x^2 + 1e-4)) / 2.0
end


function compute_camera_coverage(
    face::ActorFace,
    heading::Vector,
    distance::Vector
)
    alpha = 1.0
    face_normal = face.normal


    # current_pixel_density =
    #     alpha *
    #     abs(dot(distance, heading)) *
    #     -dot(distance, face_normal) *
    #     isvisible(distance, face_normal) / norm(distance)^4

    current_pixel_density =
        alpha *
        sqrt(dot(distance, heading)^2 + 1e-4) *
        isvisible(distance, face_normal) /
        (dot(distance, distance)^2 + 1e-4)  #(norm(distance)^2)^2) = dot(distance,distance)^2 
end

function face_view_quality(face::ActorFace, coverage_value)
    face.weight * face.area * sqrt(coverage_value + 1e-6)
end


function pinhole_ppa_reward(
    face::ActorFace,
    face_pos::Vector{Float64},  # world-frame face center (precomputed constant)
    n_world::Vector{Float64},   # world-frame face normal (precomputed constant)
    drone_x, drone_y, drone_z, drone_yaw;  # JuMP variables or plain Float64
    focal_length::Float64 = 1.2,
    tilt::Float64         = -0.35          # camera tilt (matches FPV_TILT)
)
    dx = face_pos[1] - drone_x
    dy = face_pos[2] - drone_y
    dz = face_pos[3] - drone_z

    # ─ Option A: optical-axis depth cx (yaw + tilt rotation) ───────────────
    bx = dx*cos(drone_yaw) + dy*sin(drone_yaw)  # body-frame forward depth
    cx = bx*cos(tilt)      + dz*sin(tilt)        # camera optical-axis depth
    cx_sq = cx^2 + 1e-6                          # ε guards against cx ≈ 0

    # Full distance d — still needed for cos(θ) = n_dot / d
    d = sqrt(dx^2 + dy^2 + dz^2 + 1e-6)

    # dot(n_world, -d_vec): positive when the face is pointing toward the drone
    n_dot = -(n_world[1]*dx + n_world[2]*dy + n_world[3]*dz)

    # Differentiable smooth max(0, n_dot) — needed for Ipopt gradient computation
    smooth_vis = (n_dot + sqrt(n_dot^2 + 1e-4)) / 2

    # PPA = face_area × cos(θ) × (f/cx)²
    #      = face_area × (smooth_vis/d) × f²/cx²
    #      = face_area × f² × smooth_vis / (d × cx²)
    return face.area * focal_length^2 * smooth_vis / (d * cx_sq)
end

 function SafeTrajectory(x_current::Vector{Float64},
                        u_current::Vector{Float64},
                        RobotParameters::RobotParameters,
                        actor_position::Vector{ActorState}
                        )

    
    
    N= RobotParameters.N
    states = 8
    control = 4
    target_dist = RobotParameters.target_distance
    
    #The optimization model starts here

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "max_iter", 500)
    set_optimizer_attribute(model, "tol", 1e-6)

    #Decision_Variables
    @variable(model, x[1:states,1:N+1])
    @variable(model, u[1:control, 1:N])
    
    #Variable Constraints



    #Initial condition is sent from the current state from the main
    @constraint(model, x[:,1] == x_current)
    #@constraint(model, u[:,1] == u_current)



    phi_max = pi/6  #max rotation for the drone- not too aggresive
    almax = tan(phi_max) * g #maximum lateral bound




    for k in 1:N

        #bounds for the control inputs
        @constraint(model, RobotParameters.u_min[1]<=u[1,k]<=RobotParameters.u_max[1])
        @constraint(model, RobotParameters.u_min[2]<=u[2,k]<=RobotParameters.u_max[2])
        @constraint(model, RobotParameters.u_min[3]<=u[3,k]<=RobotParameters.u_max[3])
        @constraint(model, RobotParameters.u_min[4]<=u[4,k]<=RobotParameters.u_max[4])

        #@constraint(model, -almax <=((u[1,k])^2 + (u[2,k])^2)<=almax)
        @constraint(model, ((u[1,k])^2 + (u[2,k])^2)<=almax^2)
        #Vector{Float64}
        #World frame rotation -for velocity
        @constraint(model, x[4,k+1] == x[4,k] + (u[1,k]*cos(x[7,k]) -u[2,k]*sin(x[7,k])) * RobotParameters.Ts) 
        @constraint(model, x[5,k+1] == x[5,k] + (u[1,k]*sin(x[7,k]) +u[2,k]*cos(x[7,k])) * RobotParameters.Ts)

        #Angular acceleration
        @constraint(model, x[6,k+1] == x[6,k] +u[3,k] *RobotParameters.Ts)


        #state definition- position

        @constraint(model, x[1,k+1] == x[1,k] + RobotParameters.Ts*x[4,k])  #with velocity
        @constraint(model, x[2,k+1] == x[2,k] + RobotParameters.Ts*x[5,k])  #with velocity
        @constraint(model, x[3,k+1] == x[3,k] + RobotParameters.Ts*x[6,k])  #with angular velocity


        #Angles definition
        @constraint(model, x[8,k+1] == x[8,k] + RobotParameters.Ts*u[4,k]) #Angular Velocity
        @constraint(model, x[7,k+1] == x[7,k] + RobotParameters.Ts*x[8,k]) #Angular Acceleration
     
    end


    #The cost calculation starts here
    

    cost = 0.0

    for k in 1:N

        actor_state = actor_position[min(k, length(actor_position))]  #Takes the min of the actor_horizon and lookahead to avoid "out of bounds" error 
        cx = actor_state.x
        cy = actor_state.y
        actor_heading = actor_state.heading


        #Relative position of the drone
        desired_angle = actor_heading 
        desired_x = cx + target_dist *cos(desired_angle)
        desired_y = cy + target_dist *sin(desired_angle)
        desired_z = 2.0


        #Robot Distance from the center
        error_position_x = x[1,k]-desired_x
        error_position_y = x[2,k]-desired_y
        error_position_z = x[3,k]-desired_z


        #approaching and staying to a target distance from the actor
        distance_from_circle = error_position_x^2 + error_position_y^2 +error_position_z^2
        cost += distance_from_circle 


        #Pointing camera to actor

        #heading error = theta - desired heading Vector{Float64}
        drone_to_actor_x = cx - x[1,k]
        drone_to_actor_y = cy - x[2,k]

        heading_error = cos(x[7,k]) * drone_to_actor_y - sin(x[7,k]) * drone_to_actor_x
        cost += heading_error^2

        #println("  MPC cost: $(round(cost, digits=2))")



        
        

        # ── Camera coverage: pinhole PPA reward ──────────────────────────────
        # Replaces the old compute_camera_coverage proxy (1/d⁴ falloff) with
        # the physically correct pinhole formula: face_area × cos(θ) × (f/d)².
        ppa_reward = 0.0
        for face in actor_state.mesh.faces
            face_pos = actor_world_face_center(actor_state.mesh, face,
                           actor_state.x, actor_state.y, actor_state.z,
                           actor_state.heading)
            n_world  = actor_world_normal(face, actor_state.heading)
            ppa_reward += face.weight * pinhole_ppa_reward(
                              face, face_pos, n_world, x[1,k], x[2,k], x[3,k], x[7,k])
        end
        cost -= ppa_reward

        #control cost
        ax_control_cost = (u[1,k])^2 
        cost += ax_control_cost
        ay_control_cost = (u[2,k])^2
        cost += ay_control_cost
        az_control_cost = (u[3,k])^2
        cost += az_control_cost
        angular_control_cost = u[4,k]^2
        cost += angular_control_cost

        # Terminal cost — applied only at the last horizon step k == N
        # Two one-sided penalties using smooth max(c, 0):
        #   1. Drone too far  → actor exits camera frame
        #   2. Drone too close → safety distance violation
        # if k == N
        #     for a_state in actor_position
        #         dist_sq = (x[1, N+1] - a_state.x)^2 +
        #                   (x[2, N+1] - a_state.y)^2 +
        #                   (x[3, N+1] - a_state.z)^2

        #         # Penalise when drone is farther than target_distance (actor exits frame)
        #         c_far  = dist_sq - target_dist^2
        #         cost  += (c_far + sqrt(c_far^2 + 1e-4)) / 2.0

        #         # Penalise when drone is closer than safety_dist (collision risk)
        #         c_near = RobotParameters.safety_dist^2 - dist_sq
        #         cost  += (c_near + sqrt(c_near^2 + 1e-4)) / 2.0
        #     end
        # end


        

    end



       

        



    

    #Terminal Cost

    # terminal_position_x = x[1,N] - target_position[1]
    # terminal_position_y = x[1,N] - target_position[2]
    # terminal_position = sqrt(terminal_position_x^2 + terminal_position_y^2)
    # terminal_error = terminal_position - RobotParameters.Target_distance
    # cost += terminal_error^2

    @objective(model,Min,cost)



    optimize!(model)

    println("""
    termination_status = $(termination_status(model))
    primal_status      = $(primal_status(model))
    objective_value    = $(objective_value(model))
    """)
    #assert_is_solved_and_feasible(model)
    stat = termination_status(model)
    if !(stat == MOI.OPTIMAL || stat == MOI.LOCALLY_SOLVED || stat == MOI.ITERATION_LIMIT)
        error("MPC solver failed: $stat")
    end
    stat == MOI.ITERATION_LIMIT && @warn "Solver hit iteration limit — using best feasible point"
    pos_opt = JuMP.value.(x)
    vel_opt = JuMP.value.(u)
    #println("pos_opt: $pos_opt")
    #println("u_opt: $vel_opt[1]\n")
    #println("u_opt: $vel_opt[2]\n")

    return pos_opt, vel_opt
 end


# ─────────────────────────────────────────────────────────────
# SafeTrajectoryMultiActor
#   Keeps BOTH actors in frame simultaneously.
#   Strategy:
#     • Desired position = midpoint of actors + FOLLOW_DIST
#       offset perpendicular to the actor-actor axis
#     • Heading error = drone yaw minus direction to midpoint
#     • Camera coverage reward = sum of PPA for both actors
# ─────────────────────────────────────────────────────────────
function SafeTrajectoryMultiActor(
    x_current::Vector{Float64},
    u_current::Vector{Float64},
    RobotParameters::RobotParameters,
    actor1_position::Vector{ActorState},
    actor2_position::Vector{ActorState}
)

    N           = RobotParameters.N
    states      = 8
    control     = 4
    target_dist = RobotParameters.target_distance

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "max_iter",    500)
    set_optimizer_attribute(model, "tol",         1e-6)

    @variable(model, x[1:states, 1:N+1])
    @variable(model, u[1:control, 1:N])

    @constraint(model, x[:, 1] == x_current)

    phi_max = pi / 6
    almax   = tan(phi_max) * g

    for k in 1:N
        @constraint(model, RobotParameters.u_min[1] <= u[1,k] <= RobotParameters.u_max[1])
        @constraint(model, RobotParameters.u_min[2] <= u[2,k] <= RobotParameters.u_max[2])
        @constraint(model, RobotParameters.u_min[3] <= u[3,k] <= RobotParameters.u_max[3])
        @constraint(model, RobotParameters.u_min[4] <= u[4,k] <= RobotParameters.u_max[4])
        @constraint(model, -almax <= (u[1,k])^2 + (u[2,k])^2 <= almax)

        # Dynamics
        @constraint(model, x[4,k+1] == x[4,k] + (u[1,k]*cos(x[7,k]) - u[2,k]*sin(x[7,k])) * RobotParameters.Ts)
        @constraint(model, x[5,k+1] == x[5,k] + (u[1,k]*sin(x[7,k]) + u[2,k]*cos(x[7,k])) * RobotParameters.Ts)
        @constraint(model, x[6,k+1] == x[6,k] + u[3,k] * RobotParameters.Ts)
        @constraint(model, x[1,k+1] == x[1,k] + RobotParameters.Ts * x[4,k])
        @constraint(model, x[2,k+1] == x[2,k] + RobotParameters.Ts * x[5,k])
        @constraint(model, x[3,k+1] == x[3,k] + RobotParameters.Ts * x[6,k])
        @constraint(model, x[8,k+1] == x[8,k] + RobotParameters.Ts * u[4,k])
        @constraint(model, x[7,k+1] == x[7,k] + RobotParameters.Ts * x[8,k])
    end

    cost = 0.0

    for k in 1:N
        a1 = actor1_position[min(k, length(actor1_position))]
        a2 = actor2_position[min(k, length(actor2_position))]

        cx1, cy1 = a1.x, a1.y
        cx2, cy2 = a2.x, a2.y

        # ── Midpoint between the two actors ──────────────────────
        mx = (cx1 + cx2) / 2.0
        my = (cy1 + cy2) / 2.0

        # ── Perpendicular offset to actor-actor axis ─────────────
        #   actor1→actor2 unit vector: (dx, dy)
        #   perpendicular (left-hand normal): (-dy, dx)
        dx = cx2 - cx1
        dy = cy2 - cy1
        sep = sqrt(dx^2 + dy^2 + 1e-6)   # smooth for Ipopt
        perp_x = -dy / sep
        perp_y =  dx / sep

        desired_x = mx + target_dist * perp_x
        desired_y = my + target_dist * perp_y
        desired_z = 2.0

        # ── Position error toward midpoint offset ─────────────────
        err_x = x[1,k] - desired_x
        err_y = x[2,k] - desired_y
        err_z = x[3,k] - desired_z
        cost += err_x^2 + err_y^2 + err_z^2

        # ── Heading: point camera at midpoint ────────────────────
        drone_to_mid_x = mx - x[1,k]
        drone_to_mid_y = my - x[2,k]
        heading_error  = cos(x[7,k]) * drone_to_mid_y - sin(x[7,k]) * drone_to_mid_x
        cost += heading_error^2

        # ── Camera coverage: pinhole PPA reward for both actors ──────────────
        for actor_state in (a1, a2)
            ppa_reward = 0.0
            for face in actor_state.mesh.faces
                face_pos = actor_world_face_center(actor_state.mesh, face,
                               actor_state.x, actor_state.y, actor_state.z,
                               actor_state.heading)
                n_world  = actor_world_normal(face, actor_state.heading)
                ppa_reward += face.weight * pinhole_ppa_reward(
                                  face, face_pos, n_world, x[1,k], x[2,k], x[3,k], x[7,k])
            end
            cost -= ppa_reward
        end

        # ── Control cost ─────────────────────────────────────────
        cost += u[1,k]^2 + u[2,k]^2 + u[3,k]^2 + u[4,k]^2
    end

    @objective(model, Min, cost)
    optimize!(model)

    println("""
    termination_status = $(termination_status(model))
    primal_status      = $(primal_status(model))
    objective_value    = $(objective_value(model))
    """)

    stat = termination_status(model)
    if !(stat == MOI.OPTIMAL || stat == MOI.LOCALLY_SOLVED || stat == MOI.ITERATION_LIMIT)
        error("SafeTrajectoryMultiActor solver failed: $stat")
    end
    stat == MOI.ITERATION_LIMIT && @warn "Solver hit iteration limit — using best feasible point"

    return JuMP.value.(x), JuMP.value.(u)
end


# ─────────────────────────────────────────────────────────────
# PlanTrajectory  — unified dispatcher
#
#   Pass 1 actor  → SafeTrajectory  (single-actor tracking)
#   Pass 2 actors → SafeTrajectoryMultiActor (both actors in frame)
#
#   Usage:
#     PlanTrajectory(x, u, params, [actor1_horizon])           # single
#     PlanTrajectory(x, u, params, [actor1_horizon, actor2_horizon])  # multi
# ─────────────────────────────────────────────────────────────
function PlanTrajectory(
    x_current::Vector{Float64},
    u_current::Vector{Float64},
    params::RobotParameters,
    actors::Vector{Vector{ActorState}}
)
    if length(actors) == 1
        return SafeTrajectory(x_current, u_current, params, actors[1])
    elseif length(actors) == 2
        return SafeTrajectoryMultiActor(x_current, u_current, params, actors[1], actors[2])
    else
        error("PlanTrajectory: unsupported number of actors ($(length(actors))). Pass 1 or 2.")
    end
end

    # function determine_phase(current_pos, target_pos, safe_distance, time_in_circle)
    #     """Determine whether robot should be approaching or circling"""
    #     distance_to_target = norm(current_pos[1:2] - target_pos[1:2])  
        
    #     if distance_to_target > safe_distance * 1.15  # 15% buffer for approach
    #         return "approach"
    #     elseif time_in_circle < 25.0  # Circle for 25 seconds
    #         return "circle"
    #     else
    #         return "finished"
    #     end
    # end




function mpc_run_simulation(
    current_position::Vector{Float64},
    params::RobotParameters,
    actor_trajectories::Array,
    actor_id::Int = 1
   
)




    #control parameters constraints


    



    az = 0.5*g #vertical acceleration
    aomega = pi/2 #angular acceleration


    params = RobotParameters(
    10,          #Finite Time Horizon
    0.2,    #Time step
    [-1.0,-1.0,-az,-aomega],  #minimum control bounds: ax,ay,az,atheta
    [1.0,1.0,az, aomega],  #maximum control bounds: ax,ay,az,atheta
    2.0,  #Desired filming distance from the actor
    1.0   #Minimum safe distance from the actor
    )

    #current_position = [2.0,3.5,0.0,0.0,0.0,0.0,0.0,0.0]
    current_control = [0.0,0.0,0.0,pi]
    trajectory = Vector{Vector{Float64}}()
    

    trajectory_1 = Float64[]
    trajectory_2 = Float64[]
    trajectory_3 = Float64[]
    
    #target_position = [5.0,5.0,0.0]
    safe_distance = 2.0
    time_vec = Float64[]
    
    # Simulation parameters
    sim_time = 60.0
    max_sim_time = 10.0
    t = 0.0
    
    


    
    #pos_opt, vel_opt = SafeTrajectory(current_position,current_velocity,params)

    #for t in 0:params.Ts:params.N
    for step in 1:sim_time

        #actor's trajectory and MPC trajectory horizon needs to be the same

        horizon_end = min((t + params.N-1), size(actor_trajectories,1))
        actor_horizon = [actor_trajectories[t,actor_id] for t in step:horizon_end]

        pos_opt, vel_opt = SafeTrajectory(current_position,current_control,params,actor_horizon)
        
        


        current_position = RobotDynamics(current_position,vel_opt,params.Ts)

        push!(trajectory, copy(current_position))
        push!(trajectory_1,current_position[1,end])
        push!(trajectory_2,current_position[2,end])
        push!(trajectory_3,current_position[3,end])
        push!(time_vec,t)
        t +=params.Ts
   
      
        
        

        

    
        
    end
    

    gr()
    anim = @animate for i in eachindex(time_vec)
        traj_1 = trajectory_1[1:i]
        traj_2 = trajectory_2[1:i]
        traj_3 = trajectory_3[1:i]
        
        plot(traj_1, traj_2, traj_3, linecolor = :blue)
        plot!(xlims = (-5, 5), xticks = -5:0.5:5)
        plot!(ylims = (-5, 5), yticks = -5:0.5:5)
        plot!(zlims = (-5, 5), zticks = -5:0.5:5)

        scatter!(traj_1,traj_2,traj_3,
                markersize = 0.5,
                aspect_ratio =1
                )

        annotate!(-2.25, 2.5, 0.0, "time= $(rpad(round(time_vec[i]; digits = 2), 4, "0")) s")
    end
    #drawTargets()
    gif(anim,"SafeRobot.gif", fps = 10)
    #display(plt)


    return trajectory
end

function main()

    


    
end

end