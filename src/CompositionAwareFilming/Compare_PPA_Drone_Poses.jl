# Compare_PPA_Drone_Poses.jl
# Top-down comparison of the planned drone poses for each PPA formulation, on
# the multi-actor setups of Multi_Actor_experiments_Compositional_Preferences.jl:
#   Smooth PPA + composition      (Gaussian, placement rules, priority)
#   Original PPA + composition
#   Smooth PPA, no composition    (safety + both actors fully in frame only)
#   Original PPA, no composition
# One panel per experiment; smooth PPA = blue ring, original PPA = orange dot. Saves
# drone_experiments/Compare_PPA_Drone_Poses.png and prints a table of poses.

# Include the multi-actor setups and placement rules without running its
# experiments (the skip flag is reset afterwards, so a later direct run of that
# file in the same session still runs).
global SKIP_MULTI_ACTOR_RUN = true
try
    include(joinpath(@__DIR__, "Multi_Actor_experiments_Compositional_Preferences.jl"))
finally
    global SKIP_MULTI_ACTOR_RUN = false
end

# Reward without any compositional rule: plain PPA_1 + PPA_2, shared
# constraints only (safety, both actors fully in frame).
function no_composition_score(pose)
    n = length(pose.uv)
    (pose.safe && all(pose.infull)) || return placement_fail(n)
    scores = Tuple(pose.ppa)
    return (objective=sum(scores), ok=true, assign=ntuple(_ -> 0, n), scores=scores, errs=ntuple(_ -> 0.0, n))
end

const FORMULATIONS = [
    (name="Smooth PPA + composition", model=:smooth, composition=true, color="#2a78d6", ms=13),
    (name="Original PPA + composition", model=:original, composition=true, color="#eb6834", ms=8),
    (name="Smooth PPA, no composition", model=:smooth, composition=false, color="#1baf7a", ms=13),
    (name="Original PPA, no composition", model=:original, composition=false, color="#eda100", ms=8),
]

# Experiments per setup: (label, marker shape, score function or nothing = no composition)
experiments_for(setup) = setup.tag == "A1PlusX" ?
    [("Exp 4: A1 on Top-Right", :star5, 4)] :
    [("Exp 1: both on bottom points", :circle, 1), ("Exp 2: A1 on, A2 near nearest point", :rect, 2),
     ("Exp 3: neither on a point", :diamond, 3)]

function collect_results()
    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)
    rows = []
    for setup in SETUPS, model in (:smooth, :original)
        PPA_MODEL[] = model
        poses = evaluate_poses(mesh, setup)
        println("$(setup.name), $(model) PPA: $(length(poses)) poses")
        for f in FORMULATIONS
            f.model === model || continue
            exps = f.composition ? experiments_for(setup) : [("No composition", :utriangle, 0)]
            for (label, shape, exp_id) in exps
                score = exp_id == 0 ? no_composition_score : (pose -> framing_score(exp_id, pose))
                best, _ = best_over_poses(poses, score)
                best.pose === nothing && continue
                push!(rows, (setup=setup, form=f, label=label, shape=shape,
                    pos=best.pose.drone_pos, yaw=best.pose.yaw, scores=best.scores))
            end
        end
    end
    PPA_MODEL[] = :smooth
    return rows
end

# One panel per experiment (small multiples). Smooth PPA = large ring,
# original PPA = small filled dot, so identical poses show as concentric marks.
function plot_comparison(rows, out_file)
    style = Dict(:smooth => (ms=16, mc=:white, msc="#2a78d6", msw=3, lw=3),
                 :original => (ms=7, mc="#eb6834", msc="#eb6834", msw=1, lw=1.5))
    groups = [("Side by Side", "Exp 1: both on bottom points"), ("Side by Side", "Exp 2: A1 on, A2 near nearest point"),
              ("Side by Side", "Exp 3: neither on a point"), ("Side by Side", "No composition"),
              ("Actor 1 Facing +x", "Exp 4: A1 on Top-Right"), ("Actor 1 Facing +x", "No composition")]
    panels = map(groups) do (sname, label)
        setup = only(filter(s -> s.name == sname, SETUPS))
        rs = filter(r -> r.setup.name == sname && r.label == label, rows)
        p = plot(title="$sname — $label", titlefontsize=10, xlabel="X (m)", ylabel="Y (m)",
            aspect_ratio=:equal, xlims=(-3.5, 5.5), ylims=(-4.5, 3.8), framestyle=:box,
            legend=false, gridalpha=0.15)
        for (i, a) in enumerate(setup.xy)
            scatter!(p, [a[1]], [a[2]], ms=8, mc=:gray25, msc=:white, msw=2)
            annotate!(p, a[1] - 0.35, a[2], text("A$i", :gray25, :right, 9))
            h = setup.fixed_heading[i]
            h === nothing || quiver!(p, [a[1]], [a[2]], quiver=([0.8 * cos(h)], [0.8 * sin(h)]), color=:gray25, lw=2)
        end
        for model in (:smooth, :original)
            for r in filter(r -> r.form.model === model, rs)
                st = style[model]
                x, y, _ = r.pos
                quiver!(p, [x], [y], quiver=([0.9 * cos(r.yaw)], [0.9 * sin(r.yaw)]), color=st.msc, lw=st.lw)
                scatter!(p, [x], [y], ms=st.ms, mc=st.mc, msc=st.msc, msw=st.msw)
            end
        end
        # Pose labels: one line if both formulations agree, otherwise one per formulation
        poses = [(r.form.model, r.pos, r.yaw) for r in rs]
        same = length(poses) == 2 && poses[1][2] == poses[2][2] && poses[1][3] == poses[2][3]
        lines = same ? ["smooth = original: ($(poses[1][2][1]), $(poses[1][2][2]), z=$(poses[1][2][3])), yaw $(round(rad2deg(poses[1][3]), digits=1))°"] :
                ["$(m): ($(pp[1]), $(pp[2]), z=$(pp[3])), yaw $(round(rad2deg(yw), digits=1))°" for (m, pp, yw) in poses]
        for (j, l) in enumerate(lines)
            annotate!(p, -3.3, 3.1 - 0.45 * (j - 1), text(l, :gray20, :left, 8))
        end
        p
    end
    leg = plot(framestyle=:none, grid=false, legend=:left, legendfontsize=10, xlims=(0, 1), ylims=(0, 1))
    scatter!(leg, [NaN], [NaN], ms=12, mc=:white, msc="#2a78d6", msw=3, label="Smooth PPA (ring)")
    scatter!(leg, [NaN], [NaN], ms=7, mc="#eb6834", msc="#eb6834", label="Original PPA (dot)")
    scatter!(leg, [NaN], [NaN], ms=8, mc=:gray25, msc=:white, label="Actor (arrow = fixed heading)")
    plot!(leg, [NaN], [NaN], color=:gray40, lw=2, label="Arrow at drone = camera yaw")
    p_all = plot(panels..., leg, layout=@layout([grid(2, 3) a{0.16w}]), size=(2000, 900),
        plot_title="Planned drone pose: smooth vs original PPA, with and without composition",
        plot_titlefontsize=13, left_margin=5Plots.mm, bottom_margin=9Plots.mm)
    mkpath(dirname(out_file))
    savefig(p_all, out_file)
    println("✓ Saved → $out_file")
end

function print_table(rows)
    println(rpad("Setup", 22), rpad("Formulation", 31), rpad("Experiment", 33), rpad("Drone (x, y, z)", 22), rpad("Yaw", 9), "PPA_1 / PPA_2")
    for r in rows
        x, y, z = r.pos
        println(rpad(r.setup.name, 22), rpad(r.form.name, 31), rpad(r.label, 33),
            rpad("($(x), $(y), $(z))", 22), rpad("$(round(rad2deg(r.yaw), digits=1))°", 9),
            join([string(round(s, digits=4)) for s in r.scores], " / "))
    end
end

rows = collect_results()
plot_comparison(rows, "src/CompositionAwareFilming/drone_experiments/Compare_PPA_Drone_Poses.png")
print_table(rows)
