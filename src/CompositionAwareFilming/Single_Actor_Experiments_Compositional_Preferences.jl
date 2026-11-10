# Single_Actor_Experiments_Compositional_Preferences.jl
# Rule-of-thirds framing experiments for one actor and one drone (16:9 camera).
# The camera model, view reward, constraints, drone pose search and figures are
# shared with the multi-actor experiments in ViewPlanningCore.jl; this file only
# defines the actor setup and the placement rule.
#
# One experiment per rule-of-thirds power point: the actor is fixed and faces
# the drone, and the drone pose maximizes the view reward PPA_1 with the
# Gaussian centered at that power point (σ_1), subject to the safety distance
# and the actor being fully in frame. There is no hard placement tolerance: the
# Gaussian alone pulls the actor toward the point.

# (Re)load the shared core when it is new to this session or has changed on
# disk, so edits to ViewPlanningCore.jl take effect without restarting Julia.
if !@isdefined(VIEW_PLANNING_CORE_MTIME) || VIEW_PLANNING_CORE_MTIME != mtime(joinpath(@__DIR__, "ViewPlanningCore.jl"))
    include(joinpath(@__DIR__, "ViewPlanningCore.jl"))
    global VIEW_PLANNING_CORE_MTIME = mtime(joinpath(@__DIR__, "ViewPlanningCore.jl"))
end

# Actor setup (same fields as the multi-actor setups).
const SINGLE_SETUP = (name="Single Actor", tag="Single", xy=[(-2.0, 0.0)], heading_offset=[0.0],
    fixed_heading=[nothing])

# Scores one drone pose for the experiment that puts the actor on power point k.
function single_framing_score(k::Int, pose)
    fail = placement_fail(1)
    (pose.safe && pose.infull[1]) || return fail
    return placement_result(pose, (k,))
end

function run_single_actor_composition_experiments()
    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)
    dir = "src/CompositionAwareFilming/drone_experiments"
    setup = SINGLE_SETUP
    poses = evaluate_poses(mesh, setup)
    println("$(setup.name): drone poses evaluated: $(length(poses))")
    results = [best_over_poses(poses, pose -> single_framing_score(k, pose)) for k in eachindex(POWER_POINTS)]
    clims = (0.0, maximum(maximum(g) for (_, g) in results))
    for (k, (best, grid)) in enumerate(results)
        pp = POWER_POINTS[k]
        if best.pose === nothing
            @warn "No drone pose puts the actor on the $(pp.name) power point"
            continue
        end
        render_experiment(mesh, setup, best.pose, best.assign, "Actor on the $(pp.name) Power Point",
            "Drone position: best view reward with the Gaussian on the\n$(pp.name) point (0 = no feasible pose)",
            grid, clims, placement_parts(best), "$dir/SingleActor_Composition_$(pp.tag).png")
    end
    print_placement_errors(results)
end

# Placement error of each experiment's best pose: error = desired position (the
# power point) - actor's position (its projected center), in image units
# (u ∈ [0, 2W], v ∈ [0, 2H], origin top-left).
function print_placement_errors(results)
    r3(x) = round(x, digits=3)
    println("\nPlacement error = desired position - actor's position (image units)")
    println(rpad("Experiment", 14), rpad("Desired (u, v)", 18), rpad("Actor (u, v)", 18), rpad("Error (Δu, Δv)", 20), "‖error‖")
    for (k, (best, _)) in enumerate(results)
        pp = POWER_POINTS[k]
        best.pose === nothing && (println(rpad(pp.name, 14), "no feasible pose"); continue)
        u, v = best.pose.uv[1]
        du, dv = pp.u - u, pp.v - v
        println(rpad(pp.name, 14), rpad("($(r3(pp.u)), $(r3(pp.v)))", 18), rpad("($(r3(u)), $(r3(v)))", 18),
            rpad("($(r3(du)), $(r3(dv)))", 20), r3(hypot(du, dv)))
    end
end

PPA_MODEL[] = :smooth
run_single_actor_composition_experiments()
