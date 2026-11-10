# Multi_Actor_experiments_Compositional_Preferences_OriginalPPA.jl
# Multi-actor drone pose planning with the ORIGINAL PPA and NO compositional
# rule: no power-point Gaussian, no placement rule (on/off power points) and no
# priority constraint. Uses the actor setups of
# Multi_Actor_experiments_Compositional_Preferences.jl and the shared camera,
# pose search and figures of ViewPlanningCore.jl.
#
# Original PPA (original_ppa_triangles in ViewPlanningCore.jl), per face, with
# dist = face center - drone and ĥ the drone's horizontal heading:
#   cov = |dist · ĥ| · (-dist · n) · 1[-dist · n > 0] / ‖dist‖⁴
#   PPA = Σ ω · A_face · sqrt(cov), counted only when the actor's center is in
#         the frame.
# The drone pose maximizes PPA_1 + PPA_2 subject to the safety distance and
# both actors being fully inside the image.

# Include the multi-actor setups and placement rules without running its
# experiments (the skip flag is reset afterwards, so a later direct run of that
# file in the same session still runs).
global SKIP_MULTI_ACTOR_RUN = true
try
    include(joinpath(@__DIR__, "Multi_Actor_experiments_Compositional_Preferences.jl"))
finally
    global SKIP_MULTI_ACTOR_RUN = false
end


# Shared constraints only (safety, both actors fully in frame); reward = plain
# PPA_1 + PPA_2, no composition.
function original_ppa_score(pose)
    n = length(pose.uv)
    (pose.safe && all(pose.infull)) || return placement_fail(n)
    scores = Tuple(pose.ppa)
    return (objective=sum(scores), ok=true, assign=ntuple(_ -> 0, n), scores=scores, errs=ntuple(_ -> 0.0, n))
end

function run_original_ppa_experiments()
    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)
    dir = "src/CompositionAwareFilming/drone_experiments"
    for setup in SETUPS
        poses = evaluate_poses(mesh, setup)
        println("$(setup.name): drone poses evaluated: $(length(poses))")
        best, grid = best_over_poses(poses, original_ppa_score)
        if best.pose === nothing
            @warn "No drone pose keeps both actors fully in frame ($(setup.name))"
            continue
        end
        pose = best.pose
        parts = ["A$i: PPA=$(round(best.scores[i], digits=4)), center (u,v)=($(round(pose.uv[i][1], digits=3)), $(round(pose.uv[i][2], digits=3)))"
                 for i in eachindex(best.scores)]
        render_experiment(mesh, setup, pose, best.assign, "$(setup.name): Original PPA, No Composition",
            "Drone position: best PPA₁ + PPA₂ (original PPA, no composition)\n(0 = actors not fully in frame or too close)",
            grid, (0.0, max(maximum(grid), 1e-9)), parts, "$dir/MultiActor_OriginalPPA_$(setup.tag).png";
            plain_ppa=true)
    end
end

PPA_MODEL[] = :original
try
    run_original_ppa_experiments()
finally
    PPA_MODEL[] = :smooth      # leave the session on the default (smooth) PPA
end
