module MPC


const g = 9.81

using LinearAlgebra
using Ipopt
using JuMP 
using Plots
using POMDPs
using POMDPTools


export RobotParameters

struct RobotParameters
    N::Int          #Finite Time Horizon
    Ts:: Float64    #Time step
    u_min::Vector{Float64}  # input lower bounds
    u_max::Vector{Float64}  # input upper bounds
    Dist::Float64   #Safety distance from the actor
    #target_position::Vector{Float64}
 end


#  function RobotDynamics(state,control,dt)
    
#     x,y, theta = state
#     v,omega = control

#     theta_new = theta + omega*dt
#     theta_new = atan(sin(theta_new), cos(theta_new))  #normalization

#     return [x+v*cos(theta)*dt, y+v*sin(theta)*dt, theta_new ]
#  end

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




 function SafeTrajectory(x_current::Vector{Float64},
                        u_current::Vector{Float64},
                        RobotParameters::RobotParameters,
                        actor_position::Vector{Float64}
                        )

    
    
    N= RobotParameters.N
    states = 8
    control = 4
    radius = 2.0

    #Changed the circle center to actor position
    cx,cy = actor_position[1],actor_position[2]
    


    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "max_iter", 300)
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

        @constraint(model, -almax <=((u[1,k])^2 + (u[2,k])^2)<=almax)
        #@constraint(model, ((u[1,k])^2 + (u[2,k])^2)<=almax^2)
        
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

        #Robot Distance from the centere
        error_position_x = x[1,k]-cx
        error_position_y = x[2,k]-cy


        #approaching and staying on the circle cost -Circular cost
        distance_from_circle = error_position_x^2 + error_position_y^2 -radius^2
        cost += distance_from_circle^2 



        #heading error = theta - desired heading 
        drone_to_actor_x = cx - x[1,k]
        drone_to_actor_y = cy - x[1,k]
        # desired_heading = atan(drone_to_actor_y,drone_to_actor_x)
        # heading_error = x[7,k] - desired_heading
        # heading_error = atan(sin(heading_error), cos(heading_error))

        heading_x = cos(x[7,k])
        heading_y = sin(x[7,k])

        heading_cross_product = heading_x*drone_to_actor_y - heading_y *drone_to_actor_x

        cost += heading_cross_product^2
        
        






        #control cost
        ax_control_cost = (u[1,k])^2 
        cost += ax_control_cost
        ay_control_cost = (u[2,k])^2
        cost += ay_control_cost
        az_control_cost = (u[3,k])^2
        cost += az_control_cost
        angular_control_cost = u[4,k]^2
        cost += angular_control_cost


        #reward costs start here compute_camera_coverage
        #reward = POMDPs.reward

        

        

        


    end



       

        



    

    #Terminal Cost

    # terminal_position_x = x[1,N] - target_position[1]
    # terminal_position_y = x[1,N] - target_position[2]
    # terminal_position = sqrt(terminal_position_x^2 + terminal_position_y^2)
    # terminal_error = terminal_position - RobotParameters.Dist
    # cost += terminal_error^2

    @objective(model,Min,cost)



    optimize!(model)

        # if termination_status(model) == MOI.LOCALLY_SOLVED || termination_status(model) == MOI.OPTIMAL
        # pos_opt = value.(x)
        # vel_opt = value.(u)
        # return pos_opt, vel_opt, true
        # else
        #     println("MPC solver failed: ", termination_status(model))
        #     return nothing, nothing, false
        #  end

    #solution_summary(model)

    println("""
    termination_status = $(termination_status(model))
    primal_status      = $(primal_status(model))
    objective_value    = $(objective_value(model))
    """)
    #assert_is_solved_and_feasible(model)
    stat = termination_status(model)
    if !(stat == MOI.OPTIMAL || stat == MOI.LOCALLY_SOLVED)
        error("MPC solver failed: $stat")
    end
    pos_opt = JuMP.value.(x)
    vel_opt = JuMP.value.(u)
    #println("pos_opt: $pos_opt")
    #println("u_opt: $vel_opt[1]\n")
    #println("u_opt: $vel_opt[2]\n")

    return pos_opt, vel_opt
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




function main()




    #control parameters constraints


    



    az = 0.5*g #vertical acceleration
    aomega = pi/2 #angular acceleration


    params = RobotParameters(
    10,          #Finite Time Horizon
    0.2,    #Time step
    [-1.0,-1.0,-az,-aomega],  #minimum control bounds: ax,ay,az,atheta
    [1.0,1.0,az, aomega],  #maximum control bounds: ax,ay,az,atheta
    2.0  #Safety distance from the actor(ay*sin(theta) + ay*cos(theta))
    #[2.5,2.5,0.0] #target position
    )

    current_position = [2.0,3.5,0.0,0.0,0.0,0.0,0.0,0.0]
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
    
    
    #Actor's position

    actor_pos = [0.0,0.0]



    #current_phase = "approach"

    # println("Starting reference-free aerial robot simulation...")
    # println("Initial position: $current_position")
    # println("Target position: $(params.target_position)")
    # println("Safe distance: $(safe_distance)")
    # println("Using reference-free MPC formulation")

    # while sim_time < max_sim_time

    #     #Determine current phase
    #     which_phase = determine_phase(current_position,params.target_position,params.safe_distance,time_in_circle)

    #     if which_phase == "finished"
    #         println("Circle done!!")
    #         break
    #     end

    #     if which_phase != current_phase
    #         println("Phase transition: $current_phase -> $which_phase at time $(round(sim_time, digits=2))s")
    #         if which_phase == "circle"
    #             time_in_circle = 0.0
    #         end(ay*sin(theta) + ay*cos(theta))
    #         current_phase = which_phase
    #     end
        
    # end

    #Time to solve the MPC formulation!!

    # phase_time = (current_phase == "circle") ? time_in_circle : 0.0
    #println("Current_position $current_position")can pi/2 be angular acceleration?

    
    #pos_opt, vel_opt = SafeTrajectory(current_position,current_velocity,params)

    #for t in 0:params.Ts:params.N
    while t<sim_time

        pos_opt, vel_opt = SafeTrajectory(current_position,current_control,params,actor_pos)
        

        #applied_velocity = vel_opt[:,1]

            # current_position = current_position + applied_velocity*params.Ts
            # current_velocity = applied_velocity

        current_position = RobotDynamics(current_position,vel_opt,params.Ts)


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
    
end
end