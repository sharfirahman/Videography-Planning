# StaticPPA_OrigVsSmooth.jl
# Static comparison image: Actor fixed at (-4, -4), heading toward drone camera at (4, 0, 2).
# Drone fixed at (4.0, 0.0, 2.0), yaw = π (looking back toward origin).
# Generates a side-by-side PNG heatmap showing Original PPA vs Smooth PPA
# evaluated over the full 2D actor-position grid, with the actor's static position marked.

ENV["GKSwstype"] = "100"

if !@isdefined(ActorMesh)
    include(joinpath(@__DIR__, "ActorMesh.jl"))
end
if !@isdefined(ActorTrajectory)
    include(joinpath(@__DIR__, "ActorTrajectory.jl"))
end

using .ActorMesh
using .ActorTrajectory
using Plots
using LinearAlgebra

# ── Reuse shoelace + PPA functions from OriginalPPAvssmooth.jl ────────────────
# 2D signed polygon area via shoelace.
function shoelace_signed_static(pts::Vector)
    n = length(pts)
    A = 0.0
    for i in 1:n
        j = mod1(i + 1, n)
        A += pts[i][1] * pts[j][2] - pts[j][1] * pts[i][2]
    end
    return A / 2.0
end

# ── ORIGINAL — reference-space dot-product formula ────────────────────────────
function isvisible_hard_static(dist::Vector{Float64}, face_normal::Vector{Float64})
    -dot(dist, face_normal) > 0.0 ? 1.0 : 0.0
end

function ppa_coverage_original_static(
    face::ActorFace,
    face_pos::Vector{Float64},
    n_world::Vector{Float64},
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
           isvisible_hard_static(dist, n_world) / d4

    return pixel_density
end

function ppa_quality_original_static(face::ActorFace, cov::Float64)
    face.weight * face.area * sqrt(max(cov, 0.0))
end

# ── SMOOTH — image-space PPA, differentiable relu ─────────────────────────────
function ppa_coverage_smooth_static(
    face::ActorFace,
    world_verts::Vector,
    drone::Vector{Float64},
    n_world::Vector{Float64};
    focal_length::Float64 = 1.2,
    tilt::Float64         = -0.35
)
    yaw = drone[7]
    uv  = Vector{Vector{Float64}}()
    w   = 1.0

    for idx in face.corner_indices
        v  = world_verts[idx]
        dx = v[1] - drone[1];  dy = v[2] - drone[2];  dz = v[3] - drone[3]
        bx =  dx * cos(yaw) + dy * sin(yaw)
        by = -dx * sin(yaw) + dy * cos(yaw)
        cx_raw =  bx * cos(tilt) + dz * sin(tilt)
        cy     =  by
        cz_raw = -bx * sin(tilt) + dz * cos(tilt)

        cx_soft = (cx_raw + sqrt(cx_raw^2 + 1e-4)) / 2.0
        w *= cx_soft / (cx_soft + 0.1)

        cx_denom = max(cx_raw, 0.1)
        u  = focal_length * cy     / cx_denom
        v_ = focal_length * cz_raw / cx_denom
        push!(uv, [u, v_])
    end

    face_pos = sum(world_verts[idx] for idx in face.corner_indices) / length(face.corner_indices)
    dist     = face_pos .- drone[1:3]
    n_dot    = -dot(n_world, dist)
    vis_soft = (n_dot + sqrt(n_dot^2 + 1e-4)) / 2.0

    A_mag = abs(shoelace_signed_static(uv))
    return w * vis_soft * A_mag
end

function ppa_quality_smooth_static(face::ActorFace, cov::Float64)
    face.weight * cov
end

# Evaluate both variants for a single actor + drone state.
function eval_ppa_static(
    actor::ActorState,
    drone::Vector{Float64};
    focal_length::Float64 = 1.2,
    tilt::Float64         = -0.35
)
    drone_pos = drone[1:3]
    drone_yaw = drone[7]

    world_verts = actor_world_vertices(actor.mesh, actor.x, actor.y, actor.z, actor.heading)

    # Original PPA: Preferred front face only
    preferred_face = actor.mesh.faces[1]
    face_pos_pref  = actor_world_face_center(actor.mesh, preferred_face, actor.x, actor.y, actor.z, actor.heading)
    n_world_pref   = actor_world_normal(preferred_face, actor.heading)

    cov_o = ppa_coverage_original_static(preferred_face, face_pos_pref, n_world_pref, drone_pos, drone_yaw)
    po    = ppa_quality_original_static(preferred_face, cov_o)

    # Smooth PPA: Summed across all faces
    ps = 0.0
    for face in actor.mesh.faces
        n_world = actor_world_normal(face, actor.heading)
        ps += ppa_quality_smooth_static(face,
                 ppa_coverage_smooth_static(face, world_verts, drone, n_world;
                                     focal_length=focal_length, tilt=tilt))
    end
    return po, ps
end


# ── GENERIC GRADIENT HEATMAP: Evaluate PPA at EVERY grid position ─────────────
function run_gradient_heatmap(;
    actor_x::Float64, actor_y::Float64, actor_z::Float64 = 0.0,
    actor_heading::Float64,
    label::String,
    out_file::String,
    title_str::String,
    grid_size::Int = 101,
    x_range::Tuple{Float64, Float64} = (-6.0, 6.0),
    y_range::Tuple{Float64, Float64} = (-6.0, 6.0)
)
    println("\n" * "="^70)
    println("Gradient Heatmap: $label")
    println("="^70)

    mkpath("src/mdma_greedy/drone_experiments")

    mesh = build_actor_mesh(
        actor_width=0.5, actor_depth=0.3, actor_height=0.8,
        front_weight=1.0, side_weight=0.5, top_weight=0.25,
        back_weight=0.2, bottom_weight=0.1
    )

    # Drone fixed at (4.0, 0.0, 2.0), yaw = π
    drone_pos = [4.0, 0.0, 2.0]
    drone_yaw = Float64(pi)
    drone_vec = [drone_pos[1], drone_pos[2], drone_pos[3], 0.0, 0.0, 0.0, drone_yaw, 0.0]

    println("  Actor Position : ($actor_x, $actor_y, $actor_z)")
    println("  Actor Heading  : $(round(rad2deg(actor_heading), digits=2))°")
    println("  Drone Position : ($(drone_pos[1]), $(drone_pos[2]), $(drone_pos[3]))")
    println("  Drone Yaw      : $(round(rad2deg(drone_yaw), digits=2))°")

    # ── Grid setup ────────────────────────────────────────────────────────────
    xs = range(x_range[1], x_range[2], length=grid_size)
    ys = range(y_range[1], y_range[2], length=grid_size)
    cmap = :turbo

    H_orig   = zeros(Float64, length(ys), length(xs))
    H_smooth = zeros(Float64, length(ys), length(xs))

    # Drone fixed at (4.0, 0.0, 2.0), yaw = π
    drone_pos = [4.0, 0.0, 2.0]
    drone_yaw = Float64(pi)
    drone_vec = [drone_pos[1], drone_pos[2], drone_pos[3], 0.0, 0.0, 0.0, drone_yaw, 0.0]

    # ── Actor varies across grid; drone is FIXED ──────────────────────────────
    println("  Computing PPA gradient grid ($(grid_size)×$(grid_size))...")
    println("  (Drone fixed at (4,0,2), actor position varies)")
    for (iy, y_a) in enumerate(ys)
        for (ix, x_a) in enumerate(xs)
            actor = ActorState(x_a, y_a, actor_z, actor_heading, mesh, 1)
            po, ps = eval_ppa_static(actor, drone_vec; focal_length=1.2, tilt=-0.35)
            H_orig[iy, ix]   = po
            H_smooth[iy, ix] = ps
        end
    end

    # ── PPA at the actor's exact position ─────────────────────────────────────
    actor_static = ActorState(actor_x, actor_y, actor_z, actor_heading, mesh, 1)
    po_static, ps_static = eval_ppa_static(actor_static, drone_vec; focal_length=1.2, tilt=-0.35)
    println("  PPA at Actor Position ($actor_x, $actor_y):")
    println("    Original PPA : $(round(po_static, digits=6))")
    println("    Smooth PPA   : $(round(ps_static, digits=6))")

    # ── Shared color scale: cap at 99th percentile ────────────────────────────
    all_vals = vcat(vec(H_orig), vec(H_smooth))
    all_nonzero = filter(v -> v > 0.0, all_vals)
    if isempty(all_nonzero)
        max_val = 1e-4
    else
        sorted = sort(all_nonzero)
        p99_idx = clamp(round(Int, 0.99 * length(sorted)), 1, length(sorted))
        max_val = sorted[p99_idx]
    end
    H_orig   = clamp.(H_orig,   0.0, max_val)
    H_smooth = clamp.(H_smooth, 0.0, max_val)

    # ── Left Panel: Original PPA ──────────────────────────────────────────────
    p_orig = heatmap(xs, ys, H_orig,
        color=cmap, clims=(0.0, max_val),
        title="Original PPA (Front Face Only)",
        xlabel="Actor X (m)", ylabel="Actor Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_orig, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:red, msc=:black, label="Drone (4,0,2)")
    quiver!(p_orig, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:red, lw=2.5, label="")
    scatter!(p_orig, [actor_x], [actor_y], shape=:circle, ms=7, mc=:white, msc=:black, label="Actor ($actor_x,$actor_y)")
    quiver!(p_orig, [actor_x], [actor_y],
        quiver=([0.8*cos(actor_heading)], [0.8*sin(actor_heading)]), color=:white, lw=2.0, label="")
    annotate!(p_orig, actor_x + 0.8, actor_y - 0.7,
        text("PPA=$(round(po_static, digits=4))", :white, :left, 8))

    # ── Right Panel: Smooth PPA ───────────────────────────────────────────────
    p_smooth = heatmap(xs, ys, H_smooth,
        color=cmap, clims=(0.0, max_val),
        title="Smooth PPA (All Faces, Pinhole Camera)",
        xlabel="Actor X (m)", ylabel="Actor Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_smooth, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:red, msc=:black, label="Drone (4,0,2)")
    quiver!(p_smooth, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:red, lw=2.5, label="")
    scatter!(p_smooth, [actor_x], [actor_y], shape=:circle, ms=7, mc=:white, msc=:black, label="Actor ($actor_x,$actor_y)")
    quiver!(p_smooth, [actor_x], [actor_y],
        quiver=([0.8*cos(actor_heading)], [0.8*sin(actor_heading)]), color=:white, lw=2.0, label="")
    annotate!(p_smooth, actor_x + 0.8, actor_y - 0.7,
        text("PPA=$(round(ps_static, digits=4))", :white, :left, 8))

    # ── Combine ───────────────────────────────────────────────────────────────
    p_combined = plot(p_orig, p_smooth, layout=(1, 2), size=(1400, 620),
        plot_title=title_str)

    savefig(p_combined, out_file)
    println("\n✓ Saved gradient heatmap → $out_file")

    return p_combined
end

# ── Run all 4 experiments ─────────────────────────────────────────────────────
drone_pos_ref = [4.0, 0.0, 2.0]

# Experiment 1: Actor at (-4,-4), facing TOWARD drone
heading_toward_1 = atan(drone_pos_ref[2] - (-4.0), drone_pos_ref[1] - (-4.0))
run_gradient_heatmap(
    actor_x=-4.0, actor_y=-4.0,
    actor_heading=heading_toward_1,
    label="Actor at (-4,-4), Heading TOWARD Drone",
    title_str="Static PPA Gradient: Actor at (-4,-4) Heading Toward Drone",
    out_file="src/mdma_greedy/drone_experiments/PPA_Static_Gradient_Toward.png"
)

# Experiment 2: Actor at (-4,-4), facing AWAY from drone
run_gradient_heatmap(
    actor_x=-4.0, actor_y=-4.0,
    actor_heading=heading_toward_1 + Float64(pi),
    label="Actor at (-4,-4), Heading AWAY from Drone",
    title_str="Static PPA Gradient: Actor at (-4,-4) Heading AWAY from Drone",
    out_file="src/mdma_greedy/drone_experiments/PPA_Static_Gradient_Away.png"
)

# Experiment 3: Actor at (6,4), out of FOV, facing TOWARD drone
heading_toward_3 = atan(drone_pos_ref[2] - 4.0, drone_pos_ref[1] - 6.0)
run_gradient_heatmap(
    actor_x=6.0, actor_y=4.0,
    actor_heading=heading_toward_3,
    label="Actor at (6,4), OUT OF FOV, Facing TOWARD Drone",
    title_str="Static PPA Gradient: Actor at (6,4) Out of FOV, Facing Toward Drone",
    out_file="src/mdma_greedy/drone_experiments/PPA_Static_Gradient_OutOfFOV.png",
    x_range=(-6.0, 8.0)
)

# Experiment 4: Actor at (6,4), out of FOV, facing AWAY from drone
run_gradient_heatmap(
    actor_x=6.0, actor_y=4.0,
    actor_heading=heading_toward_3 + Float64(pi),
    label="Actor at (6,4), OUT OF FOV, Facing AWAY from Drone",
    title_str="Static PPA Gradient: Actor at (6,4) Out of FOV, Facing AWAY",
    out_file="src/mdma_greedy/drone_experiments/PPA_Static_Gradient_OutOfFOV_Away.png",
    x_range=(-6.0, 8.0)
)

