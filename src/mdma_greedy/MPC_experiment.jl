using MDMA
using MPC

include("MPC_interface.jl")

# Setup MPC
az = 0.5 * MPC.g
aomega = π/2

mpc_params = MPC.RobotParameters(
    10, 0.2,
    [-1.0, -1.0, -az, -aomega],
    [1.0, 1.0, az, aomega],
    2.0
)


planner = MPC.RobotParameters

solution = mpc_run_experiment(
    experiment_name::String,
    path_to_experiments::String,
    planner::MPC.MPCPlanner,

    configs_from_file,
    save_solution,
    MDPState,UAVState,
    compute_camera_coverage,
    target_height,drone_height,
    cardinaldir,
    dirAngle
    )