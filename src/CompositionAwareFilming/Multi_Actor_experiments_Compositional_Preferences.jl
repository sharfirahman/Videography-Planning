# Multi_Actor_experiments_Compositional_Preferences.jl
# Rule-of-thirds framing experiments for two actors and one drone (16:9
# camera). The camera model, view reward, constraints, drone pose search and
# figures are shared with the single-actor experiments in ViewPlanningCore.jl;
# this file only defines the actor setups and the placement rules.
#
# Actor 1 is the priority actor (σ_1 > σ_2, and actor 2 must not occlude actor
# 1's spot). The drone pose maximizes PPA_1 + PPA_2 among poses that meet the
# shared constraints and the experiment's placement rule:
#   1. Both actors on the bottom power points (one on each)
#   2. Actor 1 on its preferred power point (A1_PREF_POINT, Bottom-Right);
#      actor 2 near its nearest power point (center within A2_NEAR_MAX of it)
#   3. Neither actor on a power point
#   4. Actor 1 (fixed heading +x) on the Top-Right point, actor 2 anywhere in
#      the frame (only partly visible required)
#   5. Actor 1 (fixed heading -x, facing backwards): the drone films actor 1's
#      front; placement as in experiment 2 (actor 1 on Bottom-Right, actor 2
#      near its nearest power point)

# (Re)load the shared core when it is new to this session or has changed on
# disk, so edits to ViewPlanningCore.jl take effect without restarting Julia.
if !@isdefined(VIEW_PLANNING_CORE_MTIME) || VIEW_PLANNING_CORE_MTIME != mtime(joinpath(@__DIR__, "ViewPlanningCore.jl"))
    include(joinpath(@__DIR__, "ViewPlanningCore.jl"))
    global VIEW_PLANNING_CORE_MTIME = mtime(joinpath(@__DIR__, "ViewPlanningCore.jl"))
end

# Actor setups. xy: the actors' ground positions; heading_offset: how far each
# actor is turned away from facing the drone; fixed_heading: a fixed world
# heading (rad) for an actor instead, or nothing to keep it facing the drone.
const SETUPS = [
    (name="Side by Side", tag="SideBySide", xy=[(-2.0, 2.0), (-2.0, -2.0)], heading_offset=[0.0, 0.0],
        fixed_heading=[nothing, nothing]),
    (name="Actor 1 Facing +x", tag="A1PlusX", xy=[(-2.0, 2.0), (-2.0, -2.0)], heading_offset=[0.0, 0.0],
        fixed_heading=[0.0, nothing]),     # actor 1 faces +x in the world
    (name="Actor 1 Facing -x", tag="A1MinusX", xy=[(-2.0, 2.0), (-2.0, -2.0)], heading_offset=[0.0, 0.0],
        fixed_heading=[Float64(π), nothing]),     # actor 1 faces -x in the world (backwards)
]

const BOTTOM_POINTS = [1, 2]   # Bottom-Left, Bottom-Right
const TOP_RIGHT = 4            # experiment 4: actor 1's preferred power point
const A2_NEAR_MAX = 0.1        # experiment 2: max distance of actor 2's center from its nearest power point
const A1_PREF_POINT = 2        # experiment 2: actor 1's preferred power point (2 = Bottom-Right)

off_all_points(u::Float64, v::Float64) = nearest_point_error(u, v) >= OFF_POINT_MIN

# The drone is in front of actor i: on the side the actor faces (angle between
# the actor's heading and the direction from the actor to the drone < 90°), so
# the camera films the actor's front.
function drone_in_front(setup, pose, i::Int)
    a, h = setup.xy[i], pose.headings[i]
    return cos(h) * (pose.drone_pos[1] - a[1]) + sin(h) * (pose.drone_pos[2] - a[2]) > 0
end

# Score of one pose for an experiment of a setup. Experiment 5 uses the
# placement rule of experiment 2 and also requires the drone to film actor 1's
# front.
experiment_score(setup, exp_id::Int, pose) =
    exp_id == 5 ? (drone_in_front(setup, pose, 1) ? framing_score(2, pose) : placement_fail(2)) :
    framing_score(exp_id, pose)

# Scores one drone pose for experiment exp_id: shared constraints, the
# experiment's placement rule (which assigns each actor its power point, 0 =
# none), then the priority constraint and the view reward PPA_1 + PPA_2.
function framing_score(exp_id::Int, pose)
    uv = pose.uv
    fail = placement_fail(2)
    pose.safe || return fail
    if exp_id == 4      # actor 1 fully in frame, actor 2 at least partly visible
        (pose.infull[1] && pose.ppa[2] > 0) || return fail
    else
        all(pose.infull) || return fail     # both actors fully inside the image
    end
    assign = nothing
    if exp_id == 1          # both actors on the bottom power points (one on each)
        best_err = Inf
        for p in BOTTOM_POINTS, q in BOTTOM_POINTS
            p == q && continue
            e1, e2 = point_error(uv[1]..., POWER_POINTS[p]), point_error(uv[2]..., POWER_POINTS[q])
            (e1 <= ON_POINT_MAX && e2 <= ON_POINT_MAX && e1 + e2 < best_err) || continue
            best_err, assign = e1 + e2, (p, q)
        end
    elseif exp_id == 2      # actor 1 on its preferred point (Bottom-Right); actor 2 near its nearest power point
        k1, k2 = A1_PREF_POINT, nearest_point(uv[2]...)
        (point_error(uv[1]..., POWER_POINTS[k1]) <= ON_POINT_MAX &&
         point_error(uv[2]..., POWER_POINTS[k2]) <= A2_NEAR_MAX) && (assign = (k1, k2))
    elseif exp_id == 3      # neither actor on a power point
        (off_all_points(uv[1]...) && off_all_points(uv[2]...)) && (assign = (0, 0))
    else                    # 4: actor 1 on the Top-Right power point, actor 2 anywhere
        point_error(uv[1]..., POWER_POINTS[TOP_RIGHT]) <= ON_POINT_MAX && (assign = (TOP_RIGHT, 0))
    end
    assign === nothing && return fail
    return placement_result(pose, assign)
end

# file_suffix / title_suffix tag the figures of variants (e.g. the original PPA).
function run_multi_actor_composition_experiments(; file_suffix::String="", title_suffix::String="")
    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)
    dir = "src/CompositionAwareFilming/drone_experiments"
    framings = [
        (1, "Both Actors on the Bottom Power Points", "BothOnPowerPoints_GaussianCenter"),
        (2, "Actor 1 on Bottom-Right, Actor 2 Near Its Nearest Point", "OneOnPowerPoint_GaussianCenter"),
        (3, "Neither Actor on a Power Point", "NoneOnPowerPoint"),
    ]
    fixed_framings = [(4, "Actor 1 on the Top-Right Power Point", "TopRight")]
    back_framings = [(5, "Actor 1 Front on Bottom-Right, Actor 2 Near Its Nearest Point", "A1FrontBottomRight")]
    # Experiments with the same placement rule; an experiment with no feasible
    # pose is still rendered, with the drone kept at its reference experiment's pose.
    reference_exp = Dict(5 => 2)
    best_poses = Dict{Int,Any}()
    for (si, setup) in enumerate(SETUPS)
        poses = evaluate_poses(mesh, setup)
        println("$(setup.name): drone poses evaluated: $(length(poses))")
        suffix = si == 1 ? "" : "_$(setup.tag)"          # first setup keeps the original file names
        prefix = si == 1 ? "" : "$(setup.name): "
        fs = setup.tag == "A1PlusX" ? fixed_framings : setup.tag == "A1MinusX" ? back_framings : framings
        results = [best_over_poses(poses, pose -> experiment_score(setup, exp_id, pose)) for (exp_id, _, _) in fs]
        clims = (0.0, maximum(maximum(g) for (_, g) in results))
        for ((exp_id, title, tag), (best, grid)) in zip(fs, results)
            if best.pose === nothing
                ref = get(best_poses, get(reference_exp, exp_id, 0), nothing)
                if ref === nothing
                    @warn "No drone pose satisfies framing experiment $exp_id ($title), and no reference pose to show"
                    continue
                end
                @warn "No drone pose satisfies framing experiment $exp_id ($title); showing the drone at experiment $(reference_exp[exp_id])'s pose"
                pose = evaluate_pose(mesh, setup, ref.drone_pos, ref.yaw; ix=ref.ix, iy=ref.iy)
                parts = ["A$i: nearest point $(POWER_POINTS[nearest_point(pose.uv[i]...)].name), " *
                         "d=$(round(nearest_point_error(pose.uv[i]...), digits=3)), plain PPA=$(round(pose.ppa[i], digits=4))"
                         for i in eachindex(pose.uv)]
                render_experiment(mesh, setup, pose, (A1_PREF_POINT, 0),
                    "$prefix$title — NO FEASIBLE POSE (drone kept at experiment $(reference_exp[exp_id])'s pose)$title_suffix",
                    "Drone position: no pose meets the experiment's placement\n(all 0; drone shown at the reference pose)",
                    grid, clims, parts, "$dir/MultiActor_Composition_$(tag)$(suffix)$(file_suffix).png")
                continue
            end
            best_poses[exp_id] = best.pose
            render_experiment(mesh, setup, best.pose, best.assign, "$prefix$title (GaussianCenter)$title_suffix",
                "Drone position: best View Reward with the experiment's\nplacement (0 = placement not possible)",
                grid, clims, placement_parts(best), "$dir/MultiActor_Composition_$(tag)$(suffix)$(file_suffix).png";
                main_region=(exp_id in (2, 5)))   # experiments 2, 5: show only the feasible region around the best pose
        end
    end
end

# Helper scripts set SKIP_MULTI_ACTOR_RUN = true while including this file to
# reuse its setups and placement rules without running the experiments.
if !(@isdefined(SKIP_MULTI_ACTOR_RUN) && SKIP_MULTI_ACTOR_RUN)
    PPA_MODEL[] = :smooth
    run_multi_actor_composition_experiments()
end
