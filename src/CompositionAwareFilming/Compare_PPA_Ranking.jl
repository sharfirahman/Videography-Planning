# Compare_PPA_Ranking.jl
# Original PPA vs smooth PPA on the same candidate drone poses (no composition):
# the "Side by Side" setup of Multi_Actor_experiments_Compositional_Preferences.jl,
# every pose that meets the shared constraints (safety distance, both actors
# fully in frame). The x-axis is the candidate pose index sorted by smooth PPA
# (panel a: by PPA_1 + PPA_2, panel b: each actor by its own PPA);
# 200 poses evenly spaced in that ranking are plotted. Each PPA is normalized
# by its maximum over the candidates (smooth is ~4x larger). Solid = smooth,
# dashed = original (panel a: original in red, both curves averaged over
# AVG_WINDOW neighbouring poses). The Spearman rank correlation over all candidates is
# printed. Saves drone_experiments/Compare_PPA_Ranking.png, and
# drone_experiments/Compare_PPA_Pose_Elements.png: the drone's x, y, z and yaw
# of all candidate poses ordered by their rank under each PPA.

# Include the multi-actor setups and placement rules without running its
# experiments (the skip flag is reset afterwards, so a later direct run of that
# file in the same session still runs).
global SKIP_MULTI_ACTOR_RUN = true
try
    include(joinpath(@__DIR__, "Multi_Actor_experiments_Compositional_Preferences.jl"))
finally
    global SKIP_MULTI_ACTOR_RUN = false
end

const N_PLOTTED = 200
const AVG_WINDOW = 10   # panel (a): running average over this many neighbouring poses, applied to both curves
const COLOR_TOTAL, COLOR_A1, COLOR_A2 = "#2a78d6", "#2a78d6", "#eb6834"

# Plain PPA of both actors for every candidate pose under one PPA model.
function plain_ppas(mesh, setup, model)
    PPA_MODEL[] = model
    poses = evaluate_poses(mesh, setup)
    PPA_MODEL[] = :smooth
    return poses
end

ranks(x) = invperm(sortperm(x))
spearman(a, b) = (ra = ranks(a); rb = ranks(b); n = length(a);
    1 - 6 * sum((ra .- rb) .^ 2) / (n * (n^2 - 1)))

function run_ppa_ranking_comparison(; setup=SETUPS[1])
    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)
    ps = plain_ppas(mesh, setup, :smooth)
    po = plain_ppas(mesh, setup, :original)     # same pose order as ps
    keep = [j for j in eachindex(ps) if ps[j].safe && all(ps[j].infull)]
    s1 = [ps[j].ppa[1] for j in keep]; s2 = [ps[j].ppa[2] for j in keep]
    o1 = [po[j].ppa[1] for j in keep]; o2 = [po[j].ppa[2] for j in keep]
    st, ot = s1 .+ s2, o1 .+ o2
    println("$(setup.name): $(length(ps)) poses, $(length(keep)) meet the constraints")
    println("Spearman rank correlation (smooth vs original): total $(round(spearman(st, ot), digits=4)), " *
            "A1 $(round(spearman(s1, o1), digits=4)), A2 $(round(spearman(s2, o2), digits=4))")
    println("Best pose same under both: $(argmax(st) == argmax(ot))")

    # N_PLOTTED poses evenly spaced in the ranking of the smooth value `v`
    pick(v) = sortperm(v)[round.(Int, range(1, length(v), length=N_PLOTTED))]
    idx, idx1, idx2 = pick(st), pick(s1), pick(s2)
    x = 1:N_PLOTTED
    nrm(v) = v ./ maximum(v)
    # Centered running average (window shrinks at the ends)
    runavg(v) = [sum(v[max(1, j - AVG_WINDOW ÷ 2):min(end, j + AVG_WINDOW ÷ 2 - 1)]) /
                 length(max(1, j - AVG_WINDOW ÷ 2):min(length(v), j + AVG_WINDOW ÷ 2 - 1)) for j in eachindex(v)]

    pa = plot(x, runavg(nrm(st)[idx]), color=COLOR_TOTAL, lw=2, label="smooth",
        ylabel="PPA₁ + PPA₂ (normalized)", title="(a) Total view reward (running average over $(AVG_WINDOW) poses)", titlefontsize=10)
    plot!(pa, x, runavg(nrm(ot)[idx]), color="#d6342a", lw=2, label="original")
    pb = plot(x, nrm(s1)[idx1], color=COLOR_A1, lw=2, label="PPA₁ smooth",
        ylabel="PPA (normalized)", title="(b) Per actor", titlefontsize=10)
    plot!(pb, x, nrm(o1)[idx1], color=COLOR_A1, lw=2, ls=:dash, alpha=0.7, label="PPA₁ original")
    plot!(pb, x, nrm(s2)[idx2], color=COLOR_A2, lw=2, label="PPA₂ smooth")
    plot!(pb, x, nrm(o2)[idx2], color=COLOR_A2, lw=2, ls=:dash, alpha=0.7, label="PPA₂ original")
    plot!(pa, xlabel="Candidate pose (sorted by smooth PPA₁ + PPA₂)")
    plot!(pb, xlabel="Candidate pose (sorted by each actor's smooth PPA)")
    for p in (pa, pb)
        plot!(p, framestyle=:box,
            gridalpha=0.15, legend=:topleft, legendfontsize=8, ylims=(0, 1.05), xlims=(1, N_PLOTTED))
    end
    p_all = plot(pa, pb, layout=(1, 2), size=(1300, 450), left_margin=6Plots.mm, bottom_margin=6Plots.mm,
        plot_title="Original vs smooth PPA on the same candidate poses ($(setup.name), $(N_PLOTTED) of $(length(keep)))",
        plot_titlefontsize=11)
    out = "src/CompositionAwareFilming/drone_experiments/Compare_PPA_Ranking.png"
    savefig(p_all, out)
    println("✓ Saved → $out")

    plot_pose_elements(ps, keep, st, ot)
end

# Drone pose elements (x, y, z, yaw) of ALL candidate poses, ordered by rank
# under each PPA (ascending PPA_1 + PPA_2): blue = smooth ranking, red =
# original ranking. No subsampling or averaging.
function plot_pose_elements(ps, keep, st, ot)
    wrapdeg(a) = rad2deg(atan(sin(a), cos(a)))
    elems = [("x (m)", j -> ps[j].drone_pos[1]), ("y (m)", j -> ps[j].drone_pos[2]),
             ("z (m)", j -> ps[j].drone_pos[3]), ("yaw (deg)", j -> wrapdeg(ps[j].yaw))]
    os, oo = keep[sortperm(st)], keep[sortperm(ot)]
    n = length(keep)
    panels = map(elems) do (lab, f)
        p = scatter(1:n, f.(os), ms=1.2, msw=0, mc="#2a78d6", ma=0.25, label="smooth ranking")
        scatter!(p, 1:n, f.(oo), ms=1.2, msw=0, mc="#d6342a", ma=0.25, label="original ranking")
        plot!(p, ylabel=lab, xlabel="Candidate pose rank (ascending PPA₁ + PPA₂)", framestyle=:box,
            gridalpha=0.15, legend=:topleft, legendfontsize=8, xlims=(1, n), markerstrokewidth=0)
        p
    end
    p_all = plot(panels..., layout=(2, 2), size=(1500, 850), left_margin=6Plots.mm, bottom_margin=5Plots.mm,
        plot_title="Drone pose elements vs PPA rank, all $(n) candidate poses (blue = smooth, red = original)",
        plot_titlefontsize=11)
    out = "src/CompositionAwareFilming/drone_experiments/Compare_PPA_Pose_Elements.png"
    savefig(p_all, out)
    println("✓ Saved → $out")
end

run_ppa_ranking_comparison()
