# OriginalPPAvssmooth.jl
# Evaluates Original PPA vs Smooth PPA:
#   Original PPA: Evaluated strictly for the PREFERRED FACE (Front Face).
#   Smooth PPA: Evaluated over all faces of the 3D subject mesh.
#   Fixed Drone sitting at (4.0, 0.0, 2.0) looking back at origin (+X -> -X, yaw = π).
#   Cumulative Visited Trajectory Heatmap: Only colors the grid cells visited by the actor over time!

ENV["GKSwstype"] = "100"

if !@isdefined(ActorMesh)
    include(joinpath(@__DIR__, "ActorMesh.jl"))
end
if !@isdefined(ActorTrajectory)
    include(joinpath(@__DIR__, "ActorTrajectory.jl"))
end
if !@isdefined(DroneVisualizationFPV)
    include(joinpath(@__DIR__, "DroneVisualizationFPV.jl"))
end

using .ActorMesh
using .ActorTrajectory
using .DroneVisualizationFPV
using Plots
using LinearAlgebra

# 2D signed polygon area via shoelace.
function shoelace_signed(pts::Vector)
    n = length(pts)
    A = 0.0
    for i in 1:n
        j = mod1(i + 1, n)
        A += pts[i][1] * pts[j][2] - pts[j][1] * pts[i][2]
    end
    return A / 2.0
end

# ── ORIGINAL — reference-space dot-product formula ────────────────────────────
function isvisible_hard(dist::Vector{Float64}, face_normal::Vector{Float64})
    -dot(dist, face_normal) > 0.0 ? 1.0 : 0.0
end

function ppa_coverage_original(
    face::ActorFace,
    face_pos::Vector{Float64},    # world-frame face centre
    n_world::Vector{Float64},     # world-frame face normal
    drone_pos::Vector{Float64},
    drone_yaw::Float64
)
    alpha = 1
    dist    = face_pos .- drone_pos
    d4      = norm(dist)^4
    d4 < 1e-6 && return 0.0
    heading = [cos(drone_yaw), sin(drone_yaw), 0.0]

    pixel_density = alpha * abs(dot(dist, heading)) *
           (-dot(dist, n_world))  *
           isvisible_hard(dist, n_world) / d4

    return pixel_density
end

function ppa_quality_original(face::ActorFace, cov::Float64)
    ppa_original_value = face.weight * face.area * sqrt(max(cov, 0.0))
    return ppa_original_value
end


# ── SMOOTH — image-space PPA, differentiable relu ─────────────────────────────
function ppa_coverage_smooth(
    face::ActorFace,
    world_verts::Vector,
    drone::Vector{Float64},
    n_world::Vector{Float64};
    focal_length::Float64 = 1.2,
    tilt::Float64         = -0.35
)
    yaw = drone[7]
    uv  = Vector{Vector{Float64}}()
    w   = 1.0   # combined soft-gate weight across all corners

    for idx in face.corner_indices
        v  = world_verts[idx]
        dx = v[1] - drone[1];  dy = v[2] - drone[2];  dz = v[3] - drone[3]
        bx =  dx * cos(yaw) + dy * sin(yaw)
        by = -dx * sin(yaw) + dy * cos(yaw)
        cx_raw =  bx * cos(tilt) + dz * sin(tilt)
        cy     =  by
        cz_raw = -bx * sin(tilt) + dz * cos(tilt)

        cx_soft = (cx_raw + sqrt(cx_raw^2 + 1e-4)) / 2.0   # smooth relu at cx=0
        w *= cx_soft / (cx_soft + 0.1)                     # smooth fade near camera plane

        cx_denom = max(cx_raw, 0.1)                         # near-clipping plane (0.1m)
        u  = focal_length * cy     / cx_denom
        v_ = focal_length * cz_raw / cx_denom
        push!(uv, [u, v_])
    end

    # Soft backface visibility gate: smooth relu on -dot(n_world, dist)
    face_pos = sum(world_verts[idx] for idx in face.corner_indices) / length(face.corner_indices)
    dist     = face_pos .- drone[1:3]
    n_dot    = -dot(n_world, dist)
    vis_soft = (n_dot + sqrt(n_dot^2 + 1e-4)) / 2.0

    A_mag = abs(shoelace_signed(uv))
    return w * vis_soft * A_mag
end

function ppa_quality_smooth(face::ActorFace, cov::Float64)
    face.weight * cov
end

# Evaluate both variants for a single actor + drone state.
function eval_ppa(
    actor::ActorState,
    drone::Vector{Float64};
    focal_length::Float64 = 1.2,
    tilt::Float64         = -0.35
)
    drone_pos = drone[1:3]
    drone_yaw = drone[7]

    world_verts = actor_world_vertices(actor.mesh, actor.x, actor.y, actor.z, actor.heading)

    # ── Original PPA: Evaluated strictly for the PREFERRED FACE (Front Face #1) ──
    preferred_face = actor.mesh.faces[1] # Front Face (:front)
    face_pos_pref  = actor_world_face_center(actor.mesh, preferred_face, actor.x, actor.y, actor.z, actor.heading)
    n_world_pref   = actor_world_normal(preferred_face, actor.heading)

    cov_o = ppa_coverage_original(preferred_face, face_pos_pref, n_world_pref, drone_pos, drone_yaw)
    po    = ppa_quality_original(preferred_face, cov_o)

    # ── Smooth PPA: Summed across all faces ──
    ps = 0.0
    for face in actor.mesh.faces
        n_world = actor_world_normal(face, actor.heading)
        ps += ppa_quality_smooth(face,
                 ppa_coverage_smooth(face, world_verts, drone, n_world;
                                     focal_length=focal_length, tilt=tilt))
    end
    return po, ps
end

# ── SPATIAL HEATMAP COMPUTATION: FIXED DRONE AT (4,0,2), ACTOR MOVING ON GRID (X_a, Y_a) ──
function compute_ppa_spatial_grid(
    actor_heading::Float64 = 0.0;
    mesh::ActorMeshStruct = build_actor_mesh(
        actor_width=0.5, actor_depth=0.3, actor_height=0.8,
        front_weight=1.0, side_weight=0.5, top_weight=0.25,
        back_weight=0.2, bottom_weight=0.1
    ),
    drone_pos::Vector{Float64} = [4.0, 0.0, 2.0],
    drone_yaw::Float64 = Float64(pi),
    grid_size::Int = 61,
    x_range::Tuple{Float64, Float64} = (-6.0, 6.0),
    y_range::Tuple{Float64, Float64} = (-6.0, 6.0),
    z_actor::Float64 = 0.0,
    focal_length::Float64 = 1.2,
    tilt::Float64 = -0.35
)
    xs = range(x_range[1], x_range[2], length=grid_size)
    ys = range(y_range[1], y_range[2], length=grid_size)

    H_orig   = zeros(Float64, length(ys), length(xs))
    H_smooth = zeros(Float64, length(ys), length(xs))

    drone = [drone_pos[1], drone_pos[2], drone_pos[3], 0.0, 0.0, 0.0, drone_yaw, 0.0]

    for (iy, y_a) in enumerate(ys)
        for (ix, x_a) in enumerate(xs)
            actor = ActorState(x_a, y_a, z_actor, actor_heading, mesh, 1)

            po, ps = eval_ppa(actor, drone; focal_length=focal_length, tilt=tilt)
            H_orig[iy, ix]   = po
            H_smooth[iy, ix] = ps
        end
    end

    return xs, ys, H_orig, H_smooth, drone
end

# ── CUMULATIVE VISITED TRAJECTORY HEATMAP ANIMATION (PERSISTENT PAINTED PATH) ──
function animate_visited_trajectory_ppa_heatmap(
    actor_traj::Vector,
    drone_fixed::Vector{Float64};
    anim_file::String = "src/mdma_greedy/drone_experiments/PPA_Visited_Trajectory_Heatmap_Drone4m.gif",
    fps::Int = 10,
    grid_size::Int = 101,
    x_range::Tuple{Float64, Float64} = (-6.0, 6.0),
    y_range::Tuple{Float64, Float64} = (-6.0, 6.0)
)
    num_frames = length(actor_traj)
    xs = range(x_range[1], x_range[2], length=grid_size)
    ys = range(y_range[1], y_range[2], length=grid_size)
    dx = step(xs); dy = step(ys)
    cmap = :turbo

    # Cumulative matrices initialized to NaN (unvisited cells background)
    H_accum_orig   = fill(NaN, length(ys), length(xs))
    H_accum_smooth = fill(NaN, length(ys), length(xs))

    drone = [drone_fixed[1], drone_fixed[2], drone_fixed[3], 0.0, 0.0, 0.0, drone_fixed[7], 0.0]

    anim = @animate for i in 1:num_frames
        actor = actor_traj[i]
        cur_x, cur_y = actor.x, actor.y
        heading = actor.heading

        # Evaluate PPA score for actor's state at frame i
        po, ps = eval_ppa(actor, drone; focal_length=1.2, tilt=-0.35)

        # Map actor's position (cur_x, cur_y) to grid cell indices (ix, iy)
        ix = clamp(round(Int, (cur_x - x_range[1]) / dx) + 1, 1, length(xs))
        iy = clamp(round(Int, (cur_y - y_range[1]) / dy) + 1, 1, length(ys))

        # Paint the visited grid cell (and adjacent 3x3 neighborhood for clear visual trail)
        for r_off in -1:1, c_off in -1:1
            r_idx = clamp(iy + r_off, 1, length(ys))
            c_idx = clamp(ix + c_off, 1, length(xs))
            H_accum_orig[r_idx, c_idx]   = po
            H_accum_smooth[r_idx, c_idx] = ps
        end

        max_o = max(maximum(filter(!isnan, H_accum_orig)), 1e-4)
        max_s = max(maximum(filter(!isnan, H_accum_smooth)), 1e-4)

        # Plot Left: Original PPA
        p_orig = heatmap(xs, ys, H_accum_orig,
            color=cmap, clims=(0.0, max_o),
            title="Original PPA (Visited Trajectory Path)",
            xlabel="Actor X (m)", ylabel="Actor Y (m)", aspect_ratio=:equal,
            xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box,
            nan_color=:gainsboro
        )
        scatter!(p_orig, [drone_fixed[1]], [drone_fixed[2]], shape=:rect, ms=8, mc=:red, msc=:black, label="Fixed Drone (4,0)")
        quiver!(p_orig, [drone_fixed[1]], [drone_fixed[2]], quiver=([0.9*cos(drone_fixed[7])], [0.9*sin(drone_fixed[7])]), color=:red, lw=2.5, label="")
        scatter!(p_orig, [cur_x], [cur_y], shape=:circle, ms=7, mc=:white, msc=:black, label="Actor Pos")
        quiver!(p_orig, [cur_x], [cur_y], quiver=([0.8*cos(heading)], [0.8*sin(heading)]), color=:white, lw=2.0, label="")

        # Plot Right: Smooth PPA
        p_smooth = heatmap(xs, ys, H_accum_smooth,
            color=cmap, clims=(0.0, max_s),
            title="Smooth PPA (Visited Trajectory Path)",
            xlabel="Actor X (m)", ylabel="Actor Y (m)", aspect_ratio=:equal,
            xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box,
            nan_color=:gainsboro
        )
        scatter!(p_smooth, [drone_fixed[1]], [drone_fixed[2]], shape=:rect, ms=8, mc=:red, msc=:black, label="Fixed Drone (4,0)")
        quiver!(p_smooth, [drone_fixed[1]], [drone_fixed[2]], quiver=([0.9*cos(drone_fixed[7])], [0.9*sin(drone_fixed[7])]), color=:red, lw=2.5, label="")
        scatter!(p_smooth, [cur_x], [cur_y], shape=:circle, ms=7, mc=:white, msc=:black, label="Actor Pos")
        quiver!(p_smooth, [cur_x], [cur_y], quiver=([0.8*cos(heading)], [0.8*sin(heading)]), color=:white, lw=2.0, label="")

        plot(p_orig, p_smooth, layout=(1, 2), size=(1300, 580))
    end

    println("Saving Visited Trajectory Heatmap GIF → $anim_file …")
    gif(anim, anim_file, fps=fps)
    return anim
end


function run_comparison(; num_steps=120)
    println("="^70)
    println("Original PPA vs Smooth PPA — Cumulative Visited Trajectory Heatmap")
    println("="^70)

    mkpath("src/mdma_greedy/drone_experiments")

    mesh = build_actor_mesh(
        actor_width=0.5, actor_depth=0.3, actor_height=0.8,
        front_weight=1.0, side_weight=0.5, top_weight=0.25,
        back_weight=0.2, bottom_weight=0.1
    )

    # Drone fixed at (4.0, 0.0, 2.0) looking back toward origin (yaw = π = 180°)
    drone_fixed = [4.0, 0.0, 2.0, 0.0, 0.0, 0.0, Float64(pi), 0.0]
    drone_traj  = fill(drone_fixed, num_steps)

    # Actor moving on figure-eight trajectory
    actor_traj  = figure_eight_trajectory(mesh; num_steps=num_steps, scale=3.0, angular_velocity=2π/15.0, actor_id=1)

    # ── 1. 3D World View + FPV Camera Panel Animation ─────────────────────────────
    println("\n[1/4] Rendering 3D World View + FPV Camera Panel Animation (Fixed Drone at 4.0m)...")
    out_fpv_world_gif = "src/mdma_greedy/drone_experiments/PPA_FPV_WorldView_Animation_Drone4m.gif"
    animate_drone_and_actor(actor_traj, drone_traj; anim_file=out_fpv_world_gif, fps=10)
    println("  Saved Animation GIF → $out_fpv_world_gif")

    # ── 2. Cumulative Visited Trajectory Heatmap GIF (Painted Path Persists) ───────
    println("\n[2/4] Rendering Cumulative Visited Trajectory Heatmap GIF (Painted Trail Persists)...")
    out_visited_gif = "src/mdma_greedy/drone_experiments/PPA_Visited_Trajectory_Heatmap_Drone4m.gif"
    animate_visited_trajectory_ppa_heatmap(actor_traj, drone_fixed; anim_file=out_visited_gif, fps=10)
    println("  Saved Visited Trajectory Heatmap GIF → $out_visited_gif")

    # ── 3. Static 2D Spatial Actor Position Heatmap ────────────────────────────────
    println("\n[3/4] Computing Static 2D Spatial Actor Position Heatmap...")
    xs, ys, H_spat_o, H_spat_s, drone = compute_ppa_spatial_grid(0.0; mesh=mesh, drone_pos=[4.0, 0.0, 2.0], drone_yaw=Float64(pi))

    cmap = :turbo
    max_spat_o = max(maximum(H_spat_o), 1e-4)
    max_spat_s = max(maximum(H_spat_s), 1e-4)

    p_spat_orig = heatmap(xs, ys, H_spat_o,
        color=cmap, clims=(0.0, max_spat_o),
        title="[1] Original PPA (Drone at 4.0m, π)",
        xlabel="Actor X (m)", ylabel="Actor Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_spat_orig, [4.0], [0.0], shape=:rect, ms=8, mc=:red, msc=:black, label="Fixed Drone (4,0)")
    quiver!(p_spat_orig, [4.0], [0.0], quiver=([0.9*cos(Float64(pi))], [0.9*sin(Float64(pi))]), color=:red, lw=2.5, label="")

    p_spat_smooth = heatmap(xs, ys, H_spat_s,
        color=cmap, clims=(0.0, max_spat_s),
        title="[2] Smooth PPA (Drone at 4.0m, π)",
        xlabel="Actor X (m)", ylabel="Actor Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_spat_smooth, [4.0], [0.0], shape=:rect, ms=8, mc=:red, msc=:black, label="Fixed Drone (4,0)")
    quiver!(p_spat_smooth, [4.0], [0.0], quiver=([0.9*cos(Float64(pi))], [0.9*sin(Float64(pi))]), color=:red, lw=2.5, label="")

    p_spat_pair = plot(p_spat_orig, p_spat_smooth, layout=(1, 2), size=(1300, 580))
    out_spatial_png = "src/mdma_greedy/drone_experiments/PPA_SpatialHeatmap_OrigVsSmooth_Drone4m.png"
    savefig(p_spat_pair, out_spatial_png)
    println("  Saved Spatial Actor Heatmaps → $out_spatial_png")

    # ── 4. Time-series Trajectory Comparison Plot ─────────────────────────────────
    println("\n[4/4] Generating Time-series Trajectory Plot...")
    steps = 1:num_steps
    ppa_orig_vals   = Float64[]
    ppa_smooth_vals = Float64[]
    for i in 1:num_steps
        po, ps = eval_ppa(actor_traj[i], drone_traj[i]; focal_length=1.2, tilt=-0.35)
        push!(ppa_orig_vals, po)
        push!(ppa_smooth_vals, ps)
    end
    p_ts = plot(steps, ppa_orig_vals, label="Original PPA (Preferred Face)", linewidth=2.5, color=:crimson, xlabel="Step", ylabel="Reward", title="PPA Trajectory Reward (Fixed Drone at 4.0m)")
    plot!(p_ts, steps, ppa_smooth_vals, label="Smooth PPA (All Faces)", linewidth=2.5, color=:royalblue, linestyle=:dash)
    out_ts_png = "src/mdma_greedy/drone_experiments/PPA_comparison_plot_Drone4m.png"
    savefig(p_ts, out_ts_png)
    println("  Saved Time-series Plot → $out_ts_png")

    println("\n✓ All visualizations generated successfully!")
    println("   1. 3D World View + FPV GIF        : $out_fpv_world_gif")
    println("   2. Visited Trajectory Heatmap GIF : $out_visited_gif")
    println("   3. Fixed Drone Spatial Actor PNG  : $out_spatial_png")
    println("   4. Time-series Trajectory Plot    : $out_ts_png")
end

run_comparison()
