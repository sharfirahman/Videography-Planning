module MDMA

include("./MDMA_States.jl")
include("./MDMA_Detection.jl")
include("./MDMA_SharedTypes.jl")
include("./MDMA_MultiAgent.jl")
include("./MDMA_Single_Robot.jl")
include("./MDMA_AssignmentPlanner.jl")
include("./MDMA_FormationPlanner.jl")
include("./MDMA_Render.jl")
include("./MDMA_Interface.jl")
include("./MDMA_Experiment.jl")
include("./MDMA_Evaluation.jl")
#include("./MPC_interface.jl")

# MPC.eval(:(actor_world_position = ActorTrajectory.actor_world_position))
# MPC.eval(:(actor_world_normal = ActorTrajectory.actor_world_normal))

end
