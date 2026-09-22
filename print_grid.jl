include("MPCRun.jl")
include("src/mdma_greedy/OriginalPPAvssmooth.jl")
using Plots
fig8_cmap = cgrad([:darkblue, :blue, :cyan, :yellow, :orange, :red])

h1 = heatmap(
    u_centers, v_centers, map(x -> x < 0.0 ? 1.1 : (x / max_ppa) * 0.8, H_occ_orig),
    title="Final Orig", color=fig8_cmap, clim=(0.0, 1.1), aspect_ratio=:equal,
    xlims=(-0.66, 0.66), ylims=(-0.49, 0.49)
)
h2 = heatmap(
    u_centers, v_centers, map(x -> x < 0.0 ? 1.1 : (x / max_ppa) * 0.8, H_occ_smooth),
    title="Final Smooth", color=fig8_cmap, clim=(0.0, 1.1), aspect_ratio=:equal,
    xlims=(-0.66, 0.66), ylims=(-0.49, 0.49)
)
savefig(plot(h1, h2, layout=(1,2), size=(1200, 600)), "final_grid.png")
