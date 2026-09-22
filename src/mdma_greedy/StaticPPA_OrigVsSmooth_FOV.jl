# StaticPPA_OrigVsSmooth_FOV.jl
# Static comparison image: Actor fixed at (-4, -4), heading toward drone camera at (4, 0, 2).
# Drone fixed at (4.0, 0.0, 2.0), yaw = π (looking back toward origin).
# Generates a side-by-side PNG heatmap showing Original PPA vs Smooth PPA
# evaluated over the full 2D actor-position grid, with the actor's static position marked.
#
# Uses the triangulated humanoid mesh (ActorMeshTriangulated) instead of the
# box-shaped ActorMesh: each face's weight is now a continuous, angle-dependent
# quantity — exp(-a * |θ_face - θ_pref|) — rather than a fixed per-side constant.

ENV["GKSwstype"] = "100"

if !@isdefined(ActorMeshTriangulated)
    include(joinpath(@__DIR__, "ActorMeshTriangulated.jl"))
end

using .ActorMeshTriangulated
using Plots
using LinearAlgebra
using Base.Iterators

const OBJ_PATH = joinpath(@__DIR__, "simple_human_rotated_color.obj")
const PART_DECAY = Dict(:Head_face => 2.0, :Body_face => 1.0, :Feet_face => 0.5, :Top_face => 1.5)

# Local stand-in for ActorTrajectory's ActorState, typed to the triangulated
# mesh instead of the box ActorMeshStruct.
struct TriActorState
    x::Float64
    y::Float64
    z::Float64
    heading::Float64
    mesh::TriMeshStruct
    actor_id::Int
end

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
    face::ActorMeshTriangulated.TriFace,
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

function ppa_quality_original_static(face::ActorMeshTriangulated.TriFace, cov::Float64, weight::Float64)
    weight * face.area * sqrt(max(cov, 0.0))
end

# ── SMOOTH — image-space PPA, differentiable relu ─────────────────────────────
function ppa_coverage_smooth_static(
    face::ActorMeshTriangulated.TriFace,
    world_verts::Vector,
    drone::Vector{Float64},
    n_world::Vector{Float64};
    focal_length::Float64 = 1.2,
    tilt::Float64         = -0.35,
    debug::Bool           = false
)
    yaw = drone[7]
    uv  = Vector{Vector{Float64}}()
    w   = 1.0

    # ── Self-Occlusion Check ──────────────────────────────────────────────────
    face_pos = sum(world_verts[idx] for idx in face.corner_indices) / length(face.corner_indices)
    dist     = face_pos .- drone[1:3]
    n_dot    = -dot(n_world, dist)

    is_occluded = n_dot <= 0.0

    visible_vertices = 0

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

        # FOV Condition
        if !is_occluded && cx_raw > 0.05 && abs(u) <= 1.0 && abs(v_) <= 1.0
            visible_vertices += 1
        end
    end

    # if debug
    #     println("    Face $(face.name): $visible_vertices / $(length(face.corner_indices)) vertices in FOV")
    # end

    # Partial credit for partially-visible faces instead of an all-or-nothing
    # cutoff — e.g. 2 of 4 vertices in FOV still counts the face in, at 50%.
    # w *= visible_vertices / length(face.corner_indices)

    # Include the face fully once at least 2 of its vertices are in FOV;
    # otherwise it contributes nothing.
    # w *= visible_vertices >= 2 ? 1.0 : 0.0

    # Include the face fully once at least 1 of its vertices is in FOV;
    # otherwise it contributes nothing.
    # w *= visible_vertices >= 1 ? 1.0 : 0.0

    # Require ALL of the face's vertices to be in FOV before counting it;
    # any partial visibility (1 or 2 of 3) is rejected entirely.
    w *= visible_vertices >= length(face.corner_indices) ? 1.0 : 0.0

    vis_soft = (n_dot + sqrt(n_dot^2 + 1e-4)) / 2.0

    A_mag = abs(shoelace_signed_static(uv))
    return w * vis_soft * A_mag
end

function ppa_quality_smooth_static(face::ActorMeshTriangulated.TriFace, cov::Float64, weight::Float64)
    weight * cov
end

# Evaluate both variants for a single actor + drone state.
function eval_ppa_static(
    actor::TriActorState,
    drone::Vector{Float64};
    focal_length::Float64 = 1.2,
    tilt::Float64         = -0.35,
    debug::Bool           = false
)
    drone_pos = drone[1:3]
    drone_yaw = drone[7]
    actor_pos = [actor.x, actor.y]

    # ── Original PPA: Target Detection + Sum across all faces ───────────────────
    po = 0.0

    # 1. Target Detection (Actor Center)
    actor_cx = actor.x - drone_pos[1]
    actor_cy = actor.y - drone_pos[2]
    actor_cz = (actor.z + actor.mesh.height/2) - drone_pos[3] # Approximate center height

    # Transform center to camera local frame
    bx_center =  actor_cx * cos(drone_yaw) + actor_cy * sin(drone_yaw)
    by_center = -actor_cx * sin(drone_yaw) + actor_cy * cos(drone_yaw)

    cx_raw_center =  bx_center * cos(tilt) + actor_cz * sin(tilt)
    cy_center     =  by_center
    cz_raw_center = -bx_center * sin(tilt) + actor_cz * cos(tilt)

    u_center = focal_length * cy_center / max(cx_raw_center, 0.1)
    v_center = focal_length * cz_raw_center / max(cx_raw_center, 0.1)

    # Check if target center is in FOV
    target_detected = cx_raw_center > 0.05 && abs(u_center) <= 1.0 && abs(v_center) <= 1.0

    if target_detected
        for face in actor.mesh.faces
            face_pos  = ActorMeshTriangulated.actor_world_face_center(actor.mesh, face, actor.x, actor.y, actor.z, actor.heading)
            n_world   = ActorMeshTriangulated.actor_world_normal(face, actor.heading)
            weight    = ActorMeshTriangulated.face_dynamic_weight(face, actor.heading, actor_pos, drone_pos[1:2])
            cov_o     = ppa_coverage_original_static(face, face_pos, n_world, drone_pos, drone_yaw)
            po       += ppa_quality_original_static(face, cov_o, weight)
        end
    end

    # ── Smooth PPA: Soft gating and Sum across all faces ───────────────────────
    world_verts = ActorMeshTriangulated.actor_world_vertices(actor.mesh, actor.x, actor.y, actor.z, actor.heading)
    ps = 0.0
    for face in actor.mesh.faces
        n_world = ActorMeshTriangulated.actor_world_normal(face, actor.heading)
        weight  = ActorMeshTriangulated.face_dynamic_weight(face, actor.heading, actor_pos, drone_pos[1:2])
        ps += ppa_quality_smooth_static(face, weight,
                 ppa_coverage_smooth_static(face, world_verts, drone, n_world;
                                     focal_length=focal_length, tilt=tilt, debug=debug))
    end
    return po, ps
end


# ── MAIN: Generate static comparison image (GIF-style: white bg, only actor cell colored) ──
function run_static_comparison(; global_max::Union{Nothing,Float64}=nothing)
    println("="^70)
    println("Static PPA Comparison: Actor at (-4,-4), Heading Toward Drone")
    println("="^70)

    mkpath("src/mdma_greedy/drone_experiments")

    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)

    # Drone fixed at (4.0, 0.0, 2.0), yaw = π (looking back toward origin)
    drone_pos = [4.0, 0.0, 2.0]
    drone_yaw = Float64(pi)
    drone_vec = [drone_pos[1], drone_pos[2], drone_pos[3], 0.0, 0.0, 0.0, drone_yaw, 0.0]

    # Actor fixed at (-4, -4), heading toward drone camera
    actor_x, actor_y, actor_z = -4.0, -4.0, 0.0
    actor_heading = atan(drone_pos[2] - actor_y, drone_pos[1] - actor_x)  # heading toward drone
    #actor_heading = pi/2

    println("  Actor Position : ($actor_x, $actor_y, $actor_z)")
    println("  Actor Heading  : $(round(rad2deg(actor_heading), digits=2))° (toward drone)")
    println("  Drone Position : ($(drone_pos[1]), $(drone_pos[2]), $(drone_pos[3]))")
    println("  Drone Yaw      : $(round(rad2deg(drone_yaw), digits=2))°")

    # ── Grid setup ────────────────────────────────────────────────────────────
    grid_size = 201
    x_range = (-8.0, 8.0)
    y_range = (-8.0, 8.0)
    xs = range(x_range[1], x_range[2], length=grid_size)
    ys = range(y_range[1], y_range[2], length=grid_size)

    # ── Evaluate PPA at the actor's static position ───────────────────────────
    actor_static = TriActorState(actor_x, actor_y, actor_z, actor_heading, mesh, 1)
    println("  PPA at Actor Position (-4,-4):")
    po_static, ps_static = eval_ppa_static(actor_static, drone_vec; focal_length=1.2, tilt=-0.35, debug=true)
    println("    Original PPA : $(round(po_static, digits=6))")
    # println("    Smooth PPA   : $(round(ps_p = mastatic, digits=6))")

    ppa_original = map(product(ys, xs)) do (y, x)
        actor_heading = atan(drone_pos[2] - y, drone_pos[1] - x)  # heading toward drone
        actora = TriActorState(x, y, actor_z, actor_heading, mesh, 1)

        po_static, ps_static = eval_ppa_static(actora, drone_vec; focal_length=1.2, tilt=-0.35, debug=true)
        return po_static
    end

    ppa_smooth = map(product(ys, xs)) do (y, x)
        actor_heading = atan(drone_pos[2] - y, drone_pos[1] - x)  # heading toward drone
        actora = TriActorState(x, y, actor_z, actor_heading, mesh, 1)
        po_static, ps_static = eval_ppa_static(actora, drone_vec; focal_length=1.2, tilt=-0.35, debug=true)
        return ps_static
    end

    # println(size(ppa_original))
    # println(size(ppa_smooth))

    # ── Gaussian blob centered on actor ───────────────────────────────────────
    # σ = 1.5
    # H_orig   = [po_static * exp(-((xi - actor_x)^2 + (yi - actor_y)^2) / (2σ^2)) for yi in ys, xi in xs]
    # H_smooth = [ps_static * exp(-((xi - actor_x)^2 + (yi - actor_y)^2) / (2σ^2)) for yi in ys, xi in xs]
    max_val = global_max === nothing ? maximum([ppa_original..., ppa_smooth...]) : global_max

    # indices = Iterators.product(xs, ys)
    # allx = map(first, indices)
    # ally = map(last, indices)

    # ── Left Panel: Original PPA ──────────────────────────────────────────────
    p_orig = heatmap(xs, ys, ppa_original,
        c=:jet, clims=(0.0, max_val),
        title="Original PPA",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_orig, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
    quiver!(p_orig, [drone_pos[1]], [drone_pos[2]],
      quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
    # scatter!(p_orig, [actor_x], [actor_y], shape=:circle, ms=6, mc=:white, msc=:black, label="Actor (-4,-4)")
    # quiver!(p_orig, [actor_x], [actor_y],
    #     quiver=([0.8*cos(actor_heading)], [0.8*sin(actor_heading)]), color=:white, lw=2.0, label="")
    # annotate!(p_orig, actor_x + 0.8, actor_y - 0.7,
    #     text("PPA=$(round(po_static, digits=4))", :white, :left, 8))

    # ── Right Panel: Smooth PPA ───────────────────────────────────────────────
    p_smooth = heatmap(xs, ys, ppa_smooth,
        c=:jet, clims=(0.0, max_val),
        title="Smooth PPA",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_smooth, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
    quiver!(p_smooth, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
    # scatter!(p_smooth, [actor_x], [actor_y], shape=:circle, ms=6, mc=:white, msc=:black, label="Actor (-4,-4)")
    # quiver!(p_smooth, [actor_x], [actor_y],
    #     quiver=([0.8*cos(actor_heading)], [0.8*sin(actor_heading)]), color=:white, lw=2.0, label="")
    # annotate!(p_smooth, actor_x + 0.8, actor_y - 0.7,
    #     text("PPA=$(round(ps_static, digits=4))", :white, :left, 8))

    # ── Combine into single image ─────────────────────────────────────────────
    p_combined = plot(p_orig, p_smooth, layout=(1, 2), size=(1400, 620),
        plot_title="PPA Across All Positions — Actor Always Facing Toward Drone at (4,0,2)")

    out_file = "src/mdma_greedy/drone_experiments/PPA_Static_OrigVsSmooth_ActorAt_neg4neg4.png"
    savefig(p_combined, out_file)
    println("\n✓ Saved static PPA comparison image → $out_file")

    return p_combined
end


# ── EXPERIMENT 2: Actor at (-4,-4) LOOKING AWAY from drone ────────────────────
function run_static_comparison_away(; global_max::Union{Nothing,Float64}=nothing)
    println("\n" * "="^70)
    println("Static PPA Comparison: Actor at (-4,-4), Heading AWAY from Drone")
    println("="^70)

    mkpath("src/mdma_greedy/drone_experiments")

    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)

    # Drone fixed at (4.0, 0.0, 2.0), yaw = π
    drone_pos = [4.0, 0.0, 2.0]
    drone_yaw = Float64(pi)
    drone_vec = [drone_pos[1], drone_pos[2], drone_pos[3], 0.0, 0.0, 0.0, drone_yaw, 0.0]

    # Actor fixed at (-4, -4), heading AWAY from drone (opposite direction)
    actor_x, actor_y, actor_z = -4.0, 0.0, 0.0
    heading_toward = atan(drone_pos[2] - actor_y, drone_pos[1] - actor_x)
    actor_heading = heading_toward - Float64(pi)  # 180° flip = looking away

    println("  Actor Position : ($actor_x, $actor_y, $actor_z)")
    println("  Actor Heading  : $(round(rad2deg(actor_heading), digits=2))° (AWAY from drone)")
    println("  Drone Position : ($(drone_pos[1]), $(drone_pos[2]), $(drone_pos[3]))")
    println("  Drone Yaw      : $(round(rad2deg(drone_yaw), digits=2))°")

    # ── Grid setup ────────────────────────────────────────────────────────────
    grid_size = 201
    x_range = (-8.0, 8.0)
    y_range = (-8.0, 8.0)
    xs = range(x_range[1], x_range[2], length=grid_size)
    ys = range(y_range[1], y_range[2], length=grid_size)

    # ── Evaluate PPA at the actor's static position ───────────────────────────
    actor_static = TriActorState(actor_x, actor_y, actor_z, actor_heading, mesh, 1)
    println("  PPA at Actor Position (-4,-4) [Looking Away]:")
    po_static, ps_static = eval_ppa_static(actor_static, drone_vec; focal_length=1.2, tilt=-0.35, debug=true)
    println("    Original PPA : $(round(po_static, digits=6))")
    println("    Smooth PPA   : $(round(ps_static, digits=6))")


    ppa_original = map(product(ys, xs)) do (y, x)
        actor_heading = atan(drone_pos[2] - y, drone_pos[1] - x) - pi
        actora = TriActorState(x, y, actor_z, actor_heading, mesh, 1)
        po_static, ps_static = eval_ppa_static(actora, drone_vec; focal_length=1.2, tilt=-0.35, debug=true)
        return po_static
    end

    ppa_smooth = map(product(ys, xs)) do (y, x)
        actor_heading = atan(drone_pos[2] - y, drone_pos[1] - x) - pi
        actora = TriActorState(x, y, actor_z, actor_heading, mesh, 1)
        po_static, ps_static = eval_ppa_static(actora, drone_vec; focal_length=1.2, tilt=-0.35, debug=true)
        return ps_static
    end


    max_val = global_max === nothing ? maximum([ppa_original..., ppa_smooth...]) : global_max


    # ── Left Panel: Original PPA ──────────────────────────────────────────────
    p_orig = heatmap(xs, ys, ppa_original,
        c=:jet, clims=(0.0, max_val),
        title="Original PPA = $(round(po_static, digits=4))",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_orig, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
    quiver!(p_orig, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
    # scatter!(p_orig, [actor_x], [actor_y], shape=:circle, ms=6, mc=:white, msc=:black, label="Actor (-4,-4)")
    # quiver!(p_orig, [actor_x], [actor_y],
    #     quiver=([0.8*cos(actor_heading)], [0.8*sin(actor_heading)]), color=:white, lw=2.0, label="")
    # annotate!(p_orig, actor_x + 0.8, actor_y - 0.7,
    #     text("PPA=$(round(po_static, digits=4))", :white, :left, 8))

    # ── Right Panel: Smooth PPA ───────────────────────────────────────────────
    p_smooth = heatmap(xs, ys, ppa_smooth,
        c=:jet, clims=(0.0, max_val),
        title="Smooth PPA = $(round(ps_static, digits=4))",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_smooth, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
    quiver!(p_smooth, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
    # scatter!(p_smooth, [actor_x], [actor_y], shape=:circle, ms=6, mc=:white, msc=:black, label="Actor (-4,-4)")
    # quiver!(p_smooth, [actor_x], [actor_y],
    #     quiver=([0.8*cos(actor_heading)], [0.8*sin(actor_heading)]), color=:white, lw=2.0, label="")
    # annotate!(p_smooth, actor_x + 0.8, actor_y - 0.7,
    #     text("PPA=$(round(ps_static, digits=4))", :white, :left, 8))

    # ── Combine into single image ─────────────────────────────────────────────
    p_combined = plot(p_orig, p_smooth, layout=(1, 2), size=(1400, 620),
        plot_title="PPA Across All Positions — Actor Always Facing Away From Drone at (4,0,2)")

    out_file = "src/mdma_greedy/drone_experiments/PPA_Static_OrigVsSmooth_ActorAt_neg4neg4_Away.png"
    savefig(p_combined, out_file)
    println("\n✓ Saved static PPA comparison image (Away) → $out_file")

    return p_combined
end

# ── EXPERIMENT 3: Actor BEHIND the drone camera (out of FOV) ──────────────────
function run_static_comparison_out_of_fov(; global_max::Union{Nothing,Float64}=nothing)
    println("\n" * "="^70)
    println("Static PPA Comparison: Actor at (6,4), OUT OF CAMERA FOV")
    println("="^70)

    mkpath("src/mdma_greedy/drone_experiments")

    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)

    # Drone fixed at (4.0, 0.0, 2.0), yaw = π (camera faces -X direction)
    drone_pos = [4.0, 0.0, 2.0]
    drone_yaw = Float64(pi)
    drone_vec = [drone_pos[1], drone_pos[2], drone_pos[3], 0.0, 0.0, 0.0, drone_yaw, 0.0]

    # Actor at (6, 4) — BEHIND the drone camera (X=6 > drone X=4, camera faces -X)
    actor_x, actor_y, actor_z = 6.0, 4.0, 0.0
    actor_heading = atan(drone_pos[2] - actor_y, drone_pos[1] - actor_x)  # facing toward drone

    println("  Actor Position : ($actor_x, $actor_y, $actor_z)")
    println("  Actor Heading  : $(round(rad2deg(actor_heading), digits=2))° (toward drone)")
    println("  Drone Position : ($(drone_pos[1]), $(drone_pos[2]), $(drone_pos[3]))")
    println("  Drone Yaw      : $(round(rad2deg(drone_yaw), digits=2))° (camera faces -X)")
    println("  ⚠ Actor is BEHIND the camera lens (out of FOV)")

    # ── Grid setup ────────────────────────────────────────────────────────────
    grid_size = 101
    x_range = (-6.0, 8.0)
    y_range = (-6.0, 6.0)
    xs = range(x_range[1], x_range[2], length=grid_size)
    ys = range(y_range[1], y_range[2], length=grid_size)

    # ── Evaluate PPA at the actor's static position ───────────────────────────
    actor_static = TriActorState(actor_x, actor_y, actor_z, actor_heading, mesh, 1)
    println("  PPA at Actor Position (6,4) [Out of FOV]:")
    po_static, ps_static = eval_ppa_static(actor_static, drone_vec; focal_length=1.2, tilt=-0.35, debug=true)
    println("    Original PPA : $(round(po_static, digits=6))")
    println("    Smooth PPA   : $(round(ps_static, digits=6))")

ppa_original = map(product(ys, xs)) do (y, x)
        actora = TriActorState(x, y, actor_z, actor_heading, mesh, 1)
        po_static, ps_static = eval_ppa_static(actora, drone_vec; focal_length=1.2, tilt=-0.35, debug=true)
        return po_static
    end

    ppa_smooth = map(product(ys, xs)) do (y, x)
        actora = TriActorState(x, y, actor_z, actor_heading, mesh, 1)
        po_static, ps_static = eval_ppa_static(actora, drone_vec; focal_length=1.2, tilt=-0.35, debug=true)
        return ps_static
    end


    max_val = global_max === nothing ? maximum([ppa_original..., ppa_smooth...]) : global_max

    # ── Left Panel: Original PPA ──────────────────────────────────────────────
    p_orig = heatmap(xs, ys, ppa_original,
        c=:jet, clims=(0.0, max_val),
        title="Original PPA = $(round(po_static, digits=4))",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_orig, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
    quiver!(p_orig, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
    # scatter!(p_orig, [actor_x], [actor_y], shape=:circle, ms=6, mc=:white, msc=:black, label="Actor (6,4)")
    # quiver!(p_orig, [actor_x], [actor_y],
    #     quiver=([0.8*cos(actor_heading)], [0.8*sin(actor_heading)]), color=:white, lw=2.0, label="")
    # annotate!(p_orig, actor_x - 1.5, actor_y - 0.7,
    #     text("PPA=$(round(po_static, digits=4))", :white, :left, 8))

    # ── Right Panel: Smooth PPA ───────────────────────────────────────────────
    p_smooth = heatmap(xs, ys, ppa_smooth,
        c=:jet, clims=(0.0, max_val),
        title="Smooth PPA = $(round(ps_static, digits=4))",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_smooth, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
    quiver!(p_smooth, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
    # scatter!(p_smooth, [actor_x], [actor_y], shape=:circle, ms=6, mc=:white, msc=:black, label="Actor (6,4)")
    # quiver!(p_smooth, [actor_x], [actor_y],
    #     quiver=([0.8*cos(actor_heading)], [0.8*sin(actor_heading)]), color=:white, lw=2.0, label="")
    # annotate!(p_smooth, actor_x - 1.5, actor_y - 0.7,
    #     text("PPA=$(round(ps_static, digits=4))", :white, :left, 8))

    # ── Combine into single image ─────────────────────────────────────────────
    p_combined = plot(p_orig, p_smooth, layout=(1, 2), size=(1400, 620),
        plot_title="Static PPA: Actor at (6,4) BEHIND Drone Camera (Out of FOV)")

    out_file = "src/mdma_greedy/drone_experiments/PPA_Static_OrigVsSmooth_OutOfFOV.png"
    savefig(p_combined, out_file)
    println("\n✓ Saved static PPA comparison image (Out of FOV) → $out_file")

    return p_combined
end

# ── EXPERIMENT 4: Actor BEHIND the drone camera AND FACING AWAY ───────────────
function run_static_comparison_out_of_fov_away(; global_max::Union{Nothing,Float64}=nothing)
    println("\n" * "="^70)
    println("Static PPA Comparison: Actor at (6,4), OUT OF FOV + FACING AWAY")
    println("="^70)

    mkpath("src/mdma_greedy/drone_experiments")

    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)

    # Drone fixed at (4.0, 0.0, 2.0), yaw = π (camera faces -X direction)
    drone_pos = [4.0, 0.0, 2.0]
    drone_yaw = Float64(pi)
    drone_vec = [drone_pos[1], drone_pos[2], drone_pos[3], 0.0, 0.0, 0.0, drone_yaw, 0.0]

    # Actor at (6, 4) — BEHIND camera AND facing AWAY from drone
    actor_x, actor_y, actor_z = 6.0, 0.0, 0.0
    heading_toward = atan(drone_pos[2] - actor_y, drone_pos[1] - actor_x)
    actor_heading = heading_toward + Float64(pi)  # 180° flip = looking away

    println("  Actor Position : ($actor_x, $actor_y, $actor_z)")
    println("  Actor Heading  : $(round(rad2deg(actor_heading), digits=2))° (AWAY from drone)")
    println("  Drone Position : ($(drone_pos[1]), $(drone_pos[2]), $(drone_pos[3]))")
    println("  Drone Yaw      : $(round(rad2deg(drone_yaw), digits=2))° (camera faces -X)")
    println("  ⚠ Actor is BEHIND the camera lens AND facing away")

    # ── Grid setup ────────────────────────────────────────────────────────────
    grid_size = 101
    x_range = (-6.0, 8.0)
    y_range = (-6.0, 6.0)
    xs = range(x_range[1], x_range[2], length=grid_size)
    ys = range(y_range[1], y_range[2], length=grid_size)

    # ── Evaluate PPA at the actor's static position ───────────────────────────
    actor_static = TriActorState(actor_x, actor_y, actor_z, actor_heading, mesh, 1)
    println("  PPA at Actor Position (6,4) [Out of FOV + Away]:")
    po_static, ps_static = eval_ppa_static(actor_static, drone_vec; focal_length=1.2, tilt=-0.35, debug=true)
    println("    Original PPA : $(round(po_static, digits=6))")
    println("    Smooth PPA   : $(round(ps_static, digits=6))")


    ppa_original = map(product(ys, xs)) do (y, x)
        heading_here = atan(drone_pos[2] - y, drone_pos[1] - x) + pi   # recomputed per point
        actora = TriActorState(x, y, actor_z, heading_here, mesh, 1)
        po_static, ps_static = eval_ppa_static(actora, drone_vec; focal_length=1.2, tilt=-0.35, debug=true)
        return po_static
    end

    ppa_smooth = map(product(ys, xs)) do (y, x)
        heading_here = atan(drone_pos[2] - y, drone_pos[1] - x) + pi   # recomputed per point
        actora = TriActorState(x, y, actor_z, heading_here, mesh, 1)
        po_static, ps_static = eval_ppa_static(actora, drone_vec; focal_length=1.2, tilt=-0.35, debug=true)
        return ps_static
    end


    max_val = global_max === nothing ? maximum([ppa_original..., ppa_smooth...]) : global_max

    # ── Left Panel: Original PPA ──────────────────────────────────────────────
    p_orig = heatmap(xs, ys, ppa_original,
        c=:jet, clims=(0.0, max_val),
        title="Original PPA = $(round(po_static, digits=4))",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_orig, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
    quiver!(p_orig, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
    # scatter!(p_orig, [actor_x], [actor_y], shape=:circle, ms=6, mc=:white, msc=:black, label="Actor (6,4)")
    # quiver!(p_orig, [actor_x], [actor_y],
    #     quiver=([0.8*cos(actor_heading)], [0.8*sin(actor_heading)]), color=:white, lw=2.0, label="")
    # annotate!(p_orig, actor_x - 1.5, actor_y - 0.7,
    #     text("PPA=$(round(po_static, digits=4))", :white, :left, 8))

    # ── Right Panel: Smooth PPA ───────────────────────────────────────────────
    p_smooth = heatmap(xs, ys, ppa_smooth,
        c=:jet, clims=(0.0, max_val),
        title="Smooth PPA = $(round(ps_static, digits=4))",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_smooth, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
    quiver!(p_smooth, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
    # scatter!(p_smooth, [actor_x], [actor_y], shape=:circle, ms=6, mc=:white, msc=:black, label="Actor (6,4)")
    # quiver!(p_smooth, [actor_x], [actor_y],
    #     quiver=([0.8*cos(actor_heading)], [0.8*sin(actor_heading)]), color=:white, lw=2.0, label="")
    # annotate!(p_smooth, actor_x - 1.5, actor_y - 0.7,
    #     text("PPA=$(round(ps_static, digits=4))", :white, :left, 8))

    # ── Combine into single image ─────────────────────────────────────────────
    p_combined = plot(p_orig, p_smooth, layout=(1, 2), size=(1400, 620),
        plot_title="Static PPA: Actor at (6,4) OUT OF FOV + Facing AWAY from Drone")

    out_file = "src/mdma_greedy/drone_experiments/PPA_Static_OrigVsSmooth_OutOfFOV_Away.png"
    savefig(p_combined, out_file)
    println("\n✓ Saved static PPA comparison image (Out of FOV + Away) → $out_file")

    return p_combined
end

function run_drone_spatial_gradient_heatmap(; global_max::Union{Nothing,Float64}=nothing)
    println("="^70)
    println("Spatial Gradient Heatmap: Gaussian PPA blob at actor position")
    println("="^70)

    mkpath("src/mdma_greedy/drone_experiments")

    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)

    # Drone fixed at (4, 0, 2), yaw = π
    drone_pos = [4.0, 0.0, 2.0]
    drone_yaw = Float64(pi)
    drone_vec = [drone_pos[1], drone_pos[2], drone_pos[3], 0.0, 0.0, 0.0, drone_yaw, 0.0]

    # Actor at (-4, -4), heading toward drone
    actor_x, actor_y, actor_z = -4.0, -4.0, 0.0
    actor_heading = atan(drone_pos[2] - actor_y, drone_pos[1] - actor_x)
    actor = TriActorState(actor_x, actor_y, actor_z, actor_heading, mesh, 1)

    # Compute PPA at the actor's position
    po_val, ps_val = eval_ppa_static(actor, drone_vec; focal_length=1.2, tilt=-0.35)
    println("  Original PPA at actor: $(round(po_val, digits=6))")
    println("  Smooth PPA at actor:   $(round(ps_val, digits=6))")

    # ── Grid setup ────────────────────────────────────────────────────────────
    grid_size = 201
    x_range = (-8.0, 8.0)
    y_range = (-8.0, 8.0)
    xs = range(x_range[1], x_range[2], length=grid_size)
    ys = range(y_range[1], y_range[2], length=grid_size)

    # Gaussian blob: peak = PPA value, centered on actor, σ controls spread
    σ = 1.5
    H_orig   = [po_val * exp(-((xi - actor_x)^2 + (yi - actor_y)^2) / (2σ^2)) for yi in ys, xi in xs]
    H_smooth = [ps_val * exp(-((xi - actor_x)^2 + (yi - actor_y)^2) / (2σ^2)) for yi in ys, xi in xs]

    max_val = global_max === nothing ? max(po_val, ps_val, 1e-4) : global_max

    # ── Left Panel: Original PPA ──────────────────────────────────────────────
    p_orig = heatmap(xs, ys, H_orig,
        c=:jet, clims=(0.0, max_val),
        title="Original PPA = $(round(po_val, digits=4))",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_orig, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
    quiver!(p_orig, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
    scatter!(p_orig, [actor_x], [actor_y], shape=:circle, ms=6, mc=:white, msc=:black, label="Actor (-4,-4)")
    quiver!(p_orig, [actor_x], [actor_y],
        quiver=([0.8*cos(actor_heading)], [0.8*sin(actor_heading)]), color=:white, lw=2.0, label="")
    annotate!(p_orig, actor_x + 0.8, actor_y - 0.7,
        text("PPA=$(round(po_val, digits=4))", :white, :left, 8))

    # ── Right Panel: Smooth PPA ───────────────────────────────────────────────
    p_smooth = heatmap(xs, ys, H_smooth,
        c=:jet, clims=(0.0, max_val),
        title="Smooth PPA = $(round(ps_val, digits=4))",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_smooth, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
    quiver!(p_smooth, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
    scatter!(p_smooth, [actor_x], [actor_y], shape=:circle, ms=6, mc=:white, msc=:black, label="Actor (-4,-4)")
    quiver!(p_smooth, [actor_x], [actor_y],
        quiver=([0.8*cos(actor_heading)], [0.8*sin(actor_heading)]), color=:white, lw=2.0, label="")
    annotate!(p_smooth, actor_x + 0.8, actor_y - 0.7,
        text("PPA=$(round(ps_val, digits=4))", :white, :left, 8))

    p_combined = plot(p_orig, p_smooth, layout=(1, 2), size=(1400, 620),
        plot_title="PPA Gaussian Gradient at Actor Position (-4, -4)")

    out_file = "src/mdma_greedy/drone_experiments/PPA_Spatial_Gradient_Jet.png"
    savefig(p_combined, out_file)
    println("✓ Saved → $out_file")
end

# ── COMBINED: Overlay all four experiment scenarios in one spatial map ────────
# Mirrors the four position/heading pairs used by the individual experiment
# functions above (toward/away no longer share a position, so each scenario
# gets its own distinct Gaussian blob).
function run_combined_all_experiments(; global_max::Union{Nothing,Float64}=nothing)
    println("\n" * "="^70)
    println("Combined View: All Actor Positions & Headings in One Map")
    println("="^70)

    mkpath("src/mdma_greedy/drone_experiments")

    mesh = build_tri_mesh(OBJ_PATH; part_decay=PART_DECAY)

    drone_pos = [4.0, 0.0, 2.0]
    drone_yaw = Float64(pi)
    drone_vec = [drone_pos[1], drone_pos[2], drone_pos[3], 0.0, 0.0, 0.0, drone_yaw, 0.0]

    # (actor_x, actor_y, heading_offset_deg, label) — offset is measured from
    # "facing the camera" (0°); 180° = facing away. Matches each experiment
    # function's setup for the toward/away cases.
    scenarios = [
        (-4.0, -4.0,   0.0, "(-4,-4) Toward"),
        (-4.0,  0.0, 180.0, "(-4,0) Away"),
        ( 6.0,  4.0,   0.0, "(6,4) Toward — OOF"),
        ( 6.0, -2.0, 180.0, "(6,-2) Away — OOF"),
        ( 0.0,  0.0,   0.0, "(0,0) Toward"),
        (-6.5, -6.5,   0.0, "(-6.5,-6.5) Toward"),
        (-7.0,  7.5, 180.0, "(-7,7.5) Away"),
        (-6.5,  3.0,  45.0, "(-6.5,3) 45°"),
        ( 2.6, -0.5,   0.0, "(2.6,-0.5) Partial FOV"),
    ]

    results = NamedTuple[]
    for (actor_x, actor_y, offset_deg, label) in scenarios
        heading_toward = atan(drone_pos[2] - actor_y, drone_pos[1] - actor_x)
        actor_heading  = heading_toward + deg2rad(offset_deg)
        actor = TriActorState(actor_x, actor_y, 0.0, actor_heading, mesh, 1)
        po, ps = eval_ppa_static(actor, drone_vec; focal_length=1.2, tilt=-0.35)
        println("  $label : Original=$(round(po, digits=4))  Smooth=$(round(ps, digits=4))")
        push!(results, (x=actor_x, y=actor_y, heading=actor_heading, label=label, po=po, ps=ps))
    end

    # ── Grid setup (covers all actor positions) ────────────────────────────────
    grid_size = 201
    xs = range(-8.0, 8.0, length=grid_size)
    ys = range(-8.0, 8.0, length=grid_size)
    σ = 1.5

    H_orig   = zeros(length(ys), length(xs))
    H_smooth = zeros(length(ys), length(xs))
    for r in results
        H_orig   .+= [r.po * exp(-((xi - r.x)^2 + (yi - r.y)^2) / (2σ^2)) for yi in ys, xi in xs]
        H_smooth .+= [r.ps * exp(-((xi - r.x)^2 + (yi - r.y)^2) / (2σ^2)) for yi in ys, xi in xs]
    end

    max_val = global_max === nothing ? max(maximum(H_orig), maximum(H_smooth), 1e-4) : global_max

    # ── Left Panel: Original PPA ──────────────────────────────────────────────
    p_orig = heatmap(xs, ys, H_orig,
        c=:jet, clims=(0.0, max_val),
        title="Original PPA — All Positions & Headings",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_orig, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
    quiver!(p_orig, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
    for (i, r) in enumerate(results)
        scatter!(p_orig, [r.x], [r.y], shape=:circle, ms=6, mc=:white, msc=:black, label="")
        quiver!(p_orig, [r.x], [r.y],
            quiver=([0.8*cos(r.heading)], [0.8*sin(r.heading)]), color=:white, lw=2.0,
            arrow=Plots.arrow(0.10, 0.10), label="")
        annotate!(p_orig, r.x + 0.8, r.y - 0.7,
            text("PPA$(i)=$(round(r.po, digits=4))", :black, :left, 7))
    end

    # ── Right Panel: Smooth PPA ───────────────────────────────────────────────
    p_smooth = heatmap(xs, ys, H_smooth,
        c=:jet, clims=(0.0, max_val),
        title="Smooth PPA — All Positions & Headings",
        xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
        xlims=(minimum(xs), maximum(xs)), ylims=(minimum(ys), maximum(ys)), framestyle=:box
    )
    scatter!(p_smooth, [drone_pos[1]], [drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="Drone (4,0,2)")
    quiver!(p_smooth, [drone_pos[1]], [drone_pos[2]],
        quiver=([0.9*cos(drone_yaw)], [0.9*sin(drone_yaw)]), color=:white, lw=2.5, label="")
    for (i, r) in enumerate(results)
        scatter!(p_smooth, [r.x], [r.y], shape=:circle, ms=6, mc=:white, msc=:black, label="")
        quiver!(p_smooth, [r.x], [r.y],
            quiver=([0.8*cos(r.heading)], [0.8*sin(r.heading)]), color=:white, lw=2.0,
            arrow=Plots.arrow(0.10, 0.10), label="")
        annotate!(p_smooth, r.x + 0.8, r.y - 0.7,
            text("PPA$(i)=$(round(r.ps, digits=4))", :black, :left, 7))
    end

    # ── Combine into single image ─────────────────────────────────────────────
    p_combined = plot(p_orig, p_smooth, layout=(1, 2), size=(1500, 650),
        plot_title="All Experiments Combined")

    out_file = "src/mdma_greedy/drone_experiments/PPA_Static_AllExperiments_Combined.png"
    savefig(p_combined, out_file)
    println("\n✓ Saved combined all-experiments image → $out_file")

    return p_combined
end

# ── MAIN: Run all static visualizers ──
# Shared color-scale ceiling across all four experiments below (and between each
# experiment's Original/Smooth panels), so brightness is comparable everywhere.
global_max_val = 0.25
run_static_comparison(global_max=global_max_val)
run_static_comparison_away(global_max=global_max_val)
run_static_comparison_out_of_fov(global_max=global_max_val)
run_static_comparison_out_of_fov_away(global_max=global_max_val)
# run_drone_spatial_gradient_heatmap(global_max=global_max_val)
# run_combined_all_experiments(global_max=global_max_val)
