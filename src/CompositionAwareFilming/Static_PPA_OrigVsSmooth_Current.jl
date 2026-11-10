# Static_PPA_OrigVsSmooth_Current.jl
# Updated version of the "Original PPA vs Smooth PPA" heatmaps of
# StaticPPA_OrigVsSmooth_FOV.jl, built on the current shared code
# (ViewPlanningCore.jl): standard pinhole camera (16:9, f = 1.2, gimbal pitch
# TILT), body-region weights PART_DECAY, smooth PPA with the pixel-density term
# (smooth_ppa_triangles) and the original PPA (original_ppa_triangles). No
# composition term.
#
# The drone is fixed at (4, 0, 2) with yaw π (looking toward -x); one actor is
# moved over the ground grid, facing toward the drone (top row) or away from it
# (bottom row). Each cell is that actor's plain PPA. All panels share one color
# scale (the smooth PPA is ~4-5x larger than the original). Saves
# drone_experiments/PPA_Static_OrigVsSmooth_Current.png.

# (Re)load the shared core when it is new to this session or has changed on
# disk, so edits to ViewPlanningCore.jl take effect without restarting Julia.
if !@isdefined(VIEW_PLANNING_CORE_MTIME) || VIEW_PLANNING_CORE_MTIME != mtime(joinpath(@__DIR__, "ViewPlanningCore.jl"))
    include(joinpath(@__DIR__, "ViewPlanningCore.jl"))
    global VIEW_PLANNING_CORE_MTIME = mtime(joinpath(@__DIR__, "ViewPlanningCore.jl"))
end

const STATIC_DRONE_POS = [4.0, 0.0, 2.0]
const STATIC_DRONE_YAW = Float64(π)
const STATIC_XS = range(-8.0, 8.0, length=161)
const STATIC_YS = range(-8.0, 8.0, length=161)

# Plain PPA of one actor at every ground position, for a PPA model and a heading
# offset from facing the drone (0 = toward, π = away).
function static_ppa_grid(mesh, ppa_fn, offset::Float64)
    grid = zeros(length(STATIC_YS), length(STATIC_XS))
    for (jx, x) in enumerate(STATIC_XS), (jy, y) in enumerate(STATIC_YS)
        h = facing(x, y, STATIC_DRONE_POS, offset)
        grid[jy, jx] = smooth_ppa_value(ppa_fn(mesh, x, y, 0.0, h, STATIC_DRONE_POS, STATIC_DRONE_YAW))
    end
    return grid
end

function static_panel(grid, title, clims)
    p = heatmap(STATIC_XS, STATIC_YS, grid, c=:jet, clims=clims, title=title,
        titlefontsize=10, xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=extrema(STATIC_XS), ylims=extrema(STATIC_YS), framestyle=:box)
    scatter!(p, [STATIC_DRONE_POS[1]], [STATIC_DRONE_POS[2]], shape=:rect, ms=8, mc=:white, msc=:black,
        label="Drone (4,0,2)")
    quiver!(p, [STATIC_DRONE_POS[1]], [STATIC_DRONE_POS[2]],
        quiver=([0.9 * cos(STATIC_DRONE_YAW)], [0.9 * sin(STATIC_DRONE_YAW)]), color=:white, lw=2.5, label="")
    return p
end

function run_static_ppa_comparison()
    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)
    grids = []
    for (offset, facing_label) in ((0.0, "facing toward drone"), (Float64(π), "facing away from drone"))
        po = static_ppa_grid(mesh, original_ppa_triangles, offset)
        ps = static_ppa_grid(mesh, smooth_ppa_triangles, offset)
        println("Actor $facing_label: max original PPA $(round(maximum(po), digits=4)), " *
                "max smooth PPA $(round(maximum(ps), digits=4))")
        push!(grids, (po, "View reward (plain) — actor $facing_label"), (ps, "View reward (compositional) — actor $facing_label"))
    end
    clims = (0.0, maximum(maximum(g) for (g, _) in grids))     # one shared color scale for all panels
    panels = [static_panel(g, t, clims) for (g, t) in grids]
    p_all = plot(panels..., layout=(2, 2), size=(1400, 1250), left_margin=5Plots.mm,
        plot_title="View rewards Across All Actor Positions — Drone Fixed at (4,0,2)",
        plot_titlefontsize=12)
    out = "src/CompositionAwareFilming/drone_experiments/PPA_Static_OrigVsSmooth_Current.png"
    mkpath(dirname(out))
    savefig(p_all, out)
    println("✓ Saved → $out")
end

run_static_ppa_comparison()
