using ModelPredictiveControl, ControlSystemsBase
using Plots

G = [ tf(1.90, [18,1]) tf(1.90,[18,1]);
    tf(-0.74,[8, 1])  tf(0.74, [8, 1]) ]

Ts = 2.0
model = setop!(LinModel(G,Ts), uop=[20,20], yop=[50,30])

mpc2 = LinMPC(model,Hp=10,Hc=2,Mwt=[1,1],Nwt=[0.1,0.1])
mpc2 = setconstraint!(mpc2, ymin=[48, -Inf])

mpc2 = LinMPC(model, Hp=10, Hc=2, Mwt=[1, 1], Nwt=[0.1, 0.1], nint_u=[1, 1])
mpc2 = setconstraint!(mpc2, ymin=[48, -Inf])
setstate!(model, zeros(model.nx))
u, y = model.uop, model() # or equivalently : y = evaloutput(model)
initstate!(mpc2, u, y)

function test_mpc(mpc2, model)
    N = 200
    ry, ul = [50, 30], 0
    u_data, y_data, ry_data = zeros(model.nu, N), zeros(model.ny, N), zeros(model.ny, N)
    for i = 1:N
        i == 51  && (ry = [50, 35])
        i == 101 && (ry = [54, 30])
        i == 151 && (ul = -20)
        y = model() # simulated measurements
        preparestate!(mpc2, y) # prepare mpc2 state estimate for current iteration
        u = mpc2(ry) # or equivalently : u = moveinput!(mpc2, ry)
        u_data[:,i], y_data[:,i], ry_data[:,i] = u, y, ry
        updatestate!(mpc2, u, y) # update mpc2 state estimate for next iteration
        updatestate!(model, u + [0; ul]) # update simulator with load disturbance
    end
    return u_data, y_data, ry_data
end
u_data, y_data, ry_data = test_mpc(mpc2, model)
t_data = Ts*(0:(size(y_data,2)-1))


function plot_data(t_data, u_data, y_data, ry_data)
    p1 = plot(t_data, y_data[1,:], label="meas.", ylabel="level")
    plot!(p1, t_data, ry_data[1,:], label="setpoint", linestyle=:dash, linetype=:steppost)
    plot!(p1, t_data, fill(48,size(t_data)), label="min", linestyle=:dot, linewidth=1.5)
    p2 = plot(t_data, y_data[2,:], label="meas.", legend=:topleft, ylabel="temp.")
    plot!(p2, t_data, ry_data[2,:],label="setpoint", linestyle=:dash, linetype=:steppost)
    p3 = plot(t_data,u_data[1,:],label="cold", linetype=:steppost, ylabel="flow rate")
    plot!(p3, t_data,u_data[2,:],label="hot", linetype=:steppost, xlabel="time (s)")
    return plot(p1, p2, p3, layout=(3,1))
end
p = plot_data(t_data, u_data, y_data, ry_data)
savefig(p, "mpc_results.png")  # Save to file instead of displaying
println("Plot saved to mpc_results.png")


# src/mdma_greedy/actor_trajectories.jl

"""
Actor trajectory definitions with faces (mesh-like structure)
Compatible with both standalone MPC and MDMA integration
"""

