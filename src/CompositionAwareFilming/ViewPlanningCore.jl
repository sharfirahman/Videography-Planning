# ViewPlanningCore.jl
# Shared view reward, constraints, drone pose search and figures for the
# composition-aware view planning experiments (any number of actors). Included
# by Single_Actor_Experiments_Compositional_Preferences.jl and
# Multi_Actor_experiments_Compositional_Preferences.jl, which only define their
# actor setups and placement rules.
#
# Camera: standard pinhole convention. Camera frame x_c right, y_c down, z_c
# forward (optical axis); the camera is tilted by TILT about its x-axis (gimbal
# pitch). Image origin at the top-left corner, principal point (CX, CY) = (W, H)
# at the image center: +u right, +v down, u ∈ [0, 2W], v ∈ [0, 2H].
#
# View reward of a drone pose = Σ_i PPA_i over the actors, with
#   PPA_i = Σ_faces ω · γ · D · g
#   ω = heading weight exp(-a · |θ_i - 0|) (face_dynamic_weight)
#   γ = visibility gate (face toward the camera, all 3 vertices in front of the
#       camera and inside the image)
#   D = pixel density term A_face * sqrt(A_fov / A_face) = sqrt(A_face * A_fov)
#       (A_face: 3D face area, A_fov: projected image area)
#   g = exp(-d^2 / (2σ_i^2)), Gaussian centered at the actor's preferred power
#       point (d = image distance from the face to the point; each face's
#       nearest point when the actor has none), σ_i = POINT_SIGMA[i]
# "plain PPA" is Σ ω · γ · D, without g.
#
# Constraints shared by every experiment (rejected poses score 0):
#   safety:    drone at least SAFETY_DIST from every actor, horizontally
#   framing:   actors fully inside the image (experiments may relax this)
#   placement: experiment-specific rule on the actors' centers (on a power point
#              = within ON_POINT_MAX of it, off = at least OFF_POINT_MIN from all)
#   priority:  every visible face of every non-priority actor has
#              1 - exp(-d^2 / (2σ_1^2)) >= PRIORITY_AWAY_MIN, d measured to actor
#              1's point (or the power point nearest actor 1's center)
# The drone pose (position, yaw = direction to the actors' midpoint + offset)
# is chosen by exhaustive search for the largest view reward.

ENV["GKSwstype"] = "100"

if !@isdefined(ActorMeshTriangulated)
    include(joinpath(@__DIR__, "ActorMeshTriangulated.jl"))
end
if !@isdefined(CompositionalPreferences)
    include(joinpath(@__DIR__, "Compositional_preferences.jl"))
end

using .ActorMeshTriangulated
using .CompositionalPreferences
using Plots
using LinearAlgebra

const OBJ_PATH = joinpath(@__DIR__, "simple_human_rotated_xaxis.obj")
const PART_DECAY = Dict(:Head_face => 2.0, :Body_face => 1.0, :Feet_face => 0.5, :Top_face => 1.5)

const TILT = -0.35
const FOCAL_LENGTH = 1.2
const IMG = CompositionalPreferences.IMG_16x9   # 16:9 camera image half-extents
const CX, CY = IMG.w, IMG.h    # principal point: image origin at the top-left corner, u ∈ [0, 2W], v ∈ [0, 2H]

const POINT_SIGMA = (0.2, 0.1)   # width of the power-point Gaussian per actor (actor 1, actor 2, ...), image units
const PRIORITY_AWAY_MIN = 0.5    # priority: every non-priority face needs 1 - g >= this to actor 1's point
const SAFETY_DIST = 1.5          # safety: min horizontal drone–actor distance (m)
const OFF_POINT_MIN = 0.1        # placement: min distance from every power point to count as "off"
const ON_POINT_MAX = 0.05        # placement: max distance from its power point to count as "on" it

# Rule-of-thirds power points, in image coords as drawn in the FPV panel.
const POWER_POINTS = [
    (name="Bottom-Left", tag="BottomLeft", u=2IMG.w / 3, v=4IMG.h / 3),    # origin top-left, +v is down
    (name="Bottom-Right", tag="BottomRight", u=4IMG.w / 3, v=4IMG.h / 3),
    (name="Top-Left", tag="TopLeft", u=2IMG.w / 3, v=2IMG.h / 3),
    (name="Top-Right", tag="TopRight", u=4IMG.w / 3, v=2IMG.h / 3),
]
const POINT_ABBREV = Dict("Bottom-Left" => "BL", "Bottom-Right" => "BR", "Top-Left" => "TL", "Top-Right" => "TR")

# Drone poses searched: ground position, yaw offset from the direction to the
# actors' midpoint, and altitude.
const DRONE_XS = range(-8.0, 8.0, length=65)
const DRONE_YS = range(-8.0, 8.0, length=65)
const YAW_OFFSETS = deg2rad.(-30.0:5.0:30.0)   # yaw turns around the direction to the actors' midpoint
const DRONE_ZS = 1.5:0.25:4.0                  # drone altitudes (0.25 m steps)

# Actor ground positions swept by the actor heatmaps (drone held fixed).
const ACTOR_GRID_XS = range(-8.0, 8.0, length=81)
const ACTOR_GRID_YS = range(-8.0, 8.0, length=81)

const Tri = NamedTuple{(:c, :u, :v),NTuple{3,Float64}}

struct ActorPlacement
    x::Float64
    y::Float64
    heading::Float64
end

# ── Camera projection (parameterized by drone pose) ───────────────────────────
# Returns (u, v, z_c), z_c = depth.
function project_point(v_world::Vector{Float64}, drone_pos::Vector{Float64}, drone_yaw::Float64)
    dx = v_world[1] - drone_pos[1]
    dy = v_world[2] - drone_pos[2]
    dz = v_world[3] - drone_pos[3]
    bx = dx * cos(drone_yaw) + dy * sin(drone_yaw)     # drone body frame: forward
    by = -dx * sin(drone_yaw) + dy * cos(drone_yaw)    # drone body frame: left
    z_c = bx * cos(TILT) + dz * sin(TILT)              # depth along the optical axis
    x_c = -by                                          # right
    y_c = bx * sin(TILT) - dz * cos(TILT)              # down
    z_safe = max(z_c, 0.1)
    u = FOCAL_LENGTH * x_c / z_safe + CX
    v = FOCAL_LENGTH * y_c / z_safe + CY
    return u, v, z_c
end

function project_or_nothing(v_world::Vector{Float64}, drone_pos::Vector{Float64}, drone_yaw::Float64)
    u, v_, depth = project_point(v_world, drone_pos, drone_yaw)
    depth < 0.05 && return nothing
    return (u, v_)
end

in_image_bounds(u, v_) = 0 <= u <= 2IMG.w && 0 <= v_ <= 2IMG.h
in_frame(u, v_, depth) = depth > 0.05 && in_image_bounds(u, v_)

function actor_center_uv(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64,
    drone_pos::Vector{Float64}, drone_yaw::Float64)
    return project_point([x, y, z + mesh.height / 2], drone_pos, drone_yaw)
end

function whole_mesh_in_frame(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64,
    drone_pos::Vector{Float64}, drone_yaw::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    for v in world_verts
        u, v_, depth = project_point(v, drone_pos, drone_yaw)
        in_frame(u, v_, depth) || return false
    end
    return true
end

# Inverse projection for one image point: follows the camera ray through
# (u, v) and returns the ground (x, y) where it reaches height `zc` (an
# actor's center height), or `nothing` if the ray never gets that low.
function uv_to_ground(u::Float64, v::Float64, zc::Float64, drone_pos::Vector{Float64}, yaw::Float64)
    side, up = -(u - CX) / FOCAL_LENGTH, -(v - CY) / FOCAL_LENGTH   # ray at depth 1 (side = left, up)
    bx = cos(TILT) - up * sin(TILT)                    # undo tilt
    dz = sin(TILT) + up * cos(TILT)
    dx = bx * cos(yaw) - side * sin(yaw)               # undo yaw
    dy = bx * sin(yaw) + side * cos(yaw)
    dz < 0 || return nothing
    t = (zc - drone_pos[3]) / dz
    return (drone_pos[1] + t * dx, drone_pos[2] + t * dy)
end

# ── Actor headings and drone yaw ──────────────────────────────────────────────
# Heading of an actor at (ax, ay) that faces the drone, turned by `offset`.
facing(ax::Float64, ay::Float64, drone_pos::Vector{Float64}, offset::Float64) =
    atan(drone_pos[2] - ay, drone_pos[1] - ax) + offset

# Heading of actor i in a setup: its fixed world heading, or facing the drone
# (+ heading_offset).
actor_heading(setup, i::Int, ax::Float64, ay::Float64, drone_pos::Vector{Float64}) =
    setup.fixed_heading[i] === nothing ? facing(ax, ay, drone_pos, setup.heading_offset[i]) : setup.fixed_heading[i]

# Yaw that points the drone at the midpoint of the actors at `xy`.
function midpoint_yaw(dx::Float64, dy::Float64, xy)
    mid_x = sum(a[1] for a in xy) / length(xy)
    mid_y = sum(a[2] for a in xy) / length(xy)
    return atan(mid_y - dy, mid_x - dx)
end

part_color(name::Symbol) = name == :Head_face ? :orange :
                           name == :Body_face ? :royalblue :
                           name == :Feet_face ? :seagreen : :crimson

# ── View reward (PPA) ─────────────────────────────────────────────────────────
function shoelace_signed(pts::Vector)
    n = length(pts)
    A = 0.0
    for i in 1:n
        j = mod1(i + 1, n)
        A += pts[i][1] * pts[j][2] - pts[j][1] * pts[i][2]
    end
    return A / 2.0
end

# Per-face view contributions ω · γ · D of one actor, with the image position
# (u, v) of each face's projected centroid (only faces that count).
function smooth_ppa_triangles(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64,
    drone_pos::Vector{Float64}, drone_yaw::Float64)
    actor_pos = [x, y]
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    tris = Tri[]
    c, s = cos(heading), sin(heading)
    for face in mesh.faces
        n_world = [face.normal[1] * c - face.normal[2] * s, face.normal[1] * s + face.normal[2] * c, face.normal[3]]
        weight = ActorMeshTriangulated.face_dynamic_weight(face, heading, actor_pos, drone_pos[1:2])   # ω

        face_pos = sum(world_verts[idx] for idx in face.corner_indices) / length(face.corner_indices)
        is_occ = -dot(n_world, face_pos .- drone_pos) <= 0.0     # back-facing
        uv = Vector{Vector{Float64}}()
        vis_n = 0
        for idx in face.corner_indices
            u, v_, depth = project_point(world_verts[idx], drone_pos, drone_yaw)
            push!(uv, [u, v_])
            (!is_occ && depth > 0.05 && in_image_bounds(u, v_)) && (vis_n += 1)
        end
        vis_n >= length(face.corner_indices) || continue           # γ = 0
        A_fov = abs(shoelace_signed(uv))                           # projected image area
        density = sqrt(face.area * A_fov)                          # D = A_face * sqrt(A_fov / A_face)
        contribution = weight * vis_n * density
        contribution > 0 && push!(tris, (c=contribution, u=sum(p[1] for p in uv) / 3, v=sum(p[2] for p in uv) / 3))
    end
    return tris
end

# Original PPA (reference-space formula from StaticPPA_OrigVsSmooth_FOV.jl):
# per face, with dist = face center - drone and ĥ the drone's horizontal heading,
#   cov = |dist · ĥ| · (-dist · n) · 1[-dist · n > 0] / ‖dist‖⁴
#   contribution = ω · A_face · sqrt(cov)
# counted only when the actor's center is in the frame (target detected). Each
# face's image position (u, v) is its projected center, used by the
# composition Gaussian.
function original_ppa_triangles(mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64,
    drone_pos::Vector{Float64}, drone_yaw::Float64)
    tris = Tri[]
    u_c, v_c, depth_c = actor_center_uv(mesh, x, y, z, drone_pos, drone_yaw)
    in_frame(u_c, v_c, depth_c) || return tris                  # target not detected
    actor_pos = [x, y]
    h = [cos(drone_yaw), sin(drone_yaw), 0.0]
    for face in mesh.faces
        face_pos = ActorMeshTriangulated.actor_world_face_center(mesh, face, x, y, z, heading)
        n_world = ActorMeshTriangulated.actor_world_normal(face, heading)
        weight = ActorMeshTriangulated.face_dynamic_weight(face, heading, actor_pos, drone_pos[1:2])   # ω
        dist = face_pos .- drone_pos
        d4 = norm(dist)^4
        d4 < 1e-6 && continue
        facing_cam = -dot(dist, n_world)
        facing_cam > 0.0 || continue                                # back-facing
        cov = abs(dot(dist, h)) * facing_cam / d4
        contribution = weight * face.area * sqrt(cov)
        u, v, _ = project_point(face_pos, drone_pos, drone_yaw)
        contribution > 0 && push!(tris, (c=contribution, u=u, v=v))
    end
    return tris
end

# Which PPA the experiments use: :smooth (smooth_ppa_triangles, default) or
# :original (original_ppa_triangles).
const PPA_MODEL = Ref(:smooth)
face_contributions(args...) = PPA_MODEL[] === :original ? original_ppa_triangles(args...) : smooth_ppa_triangles(args...)

# Plain PPA: the sum of all face contributions (no composition term).
smooth_ppa_value(tris) = sum((t.c for t in tris); init=0.0)

# PPA of actor i with the composition term inside, per face, measured to power
# point k (0 = each face's nearest point).
smooth_ppa_value(tris, i::Int, k::Int=0) =
    sum((t.c * artistic_term(i, (t.u, t.v), k) for t in tris); init=0.0)

point_error(u::Float64, v::Float64, p) = sqrt((p.u - u)^2 + (p.v - v)^2)
nearest_point_error(u::Float64, v::Float64) = minimum(point_error(u, v, p) for p in POWER_POINTS)
nearest_point(u::Float64, v::Float64) = argmin([point_error(u, v, p) for p in POWER_POINTS])

center_gaussian(d::Float64, σ::Float64) = exp(-d^2 / (2 * σ^2))

# Distance of (u, v) from power point k, or from the nearest one (k = 0).
point_distance(uv, k::Int) = k > 0 ? point_error(uv..., POWER_POINTS[k]) : nearest_point_error(uv...)

# Composition term of actor i: g = exp(-d^2 / (2σ_i^2)), d the distance of
# image point `uv` from power point k (or the nearest one, k = 0).
artistic_term(i::Int, uv, k::Int=0) = center_gaussian(point_distance(uv, k), POINT_SIGMA[i])

# Actor i's PPA in an evaluated pose, measured to power point k (0 = nearest);
# pose.appa[i][k + 1] holds smooth_ppa_value(tris, i, k).
actor_score(i::Int, pose, k::Int=0) = pose.appa[i][k+1]

# Every actor's PPA for a pose, actor i measured to its point assign[i].
actor_ppas(pose, assign) = Tuple(actor_score(i, pose, assign[i]) for i in eachindex(pose.appa))

# ── Priority constraint ───────────────────────────────────────────────────────
# Actor 1's spot, which the other actors must not occlude: actor 1's point k1,
# or the power point nearest actor 1's center when it has none (k1 = 0).
priority_point(uv1, k1::Int) = k1 > 0 ? k1 : nearest_point(uv1...)

# Largest g = exp(-d^2 / (2σ_1^2)) over a set of faces, d to power point k
# (actor 1's σ: it measures how far another actor stays from actor 1's spot).
max_point_g(tris, k::Int) = maximum((center_gaussian(point_error(t.u, t.v, POWER_POINTS[k]), POINT_SIGMA[1]) for t in tris); init=0.0)

# Every visible face of every non-priority actor has 1 - g >= PRIORITY_AWAY_MIN
# to actor 1's spot (always true with a single actor).
function priority_ok(pose, k1::Int=0)
    length(pose.uv) > 1 || return true
    k = priority_point(pose.uv[1], k1)
    return all(1 - pose.maxg[i][k] >= PRIORITY_AWAY_MIN for i in 2:length(pose.uv))
end

# ── Pose evaluation and search ────────────────────────────────────────────────
# Evaluates one drone pose for an actor setup: each actor's heading, plain PPA,
# PPA with the composition term (for every power point choice), image center,
# visibility flags, and the safety check.
function evaluate_pose(mesh::TriMeshStruct, setup, drone_pos::Vector{Float64}, yaw::Float64; ix::Int=0, iy::Int=0)
    xy = setup.xy
    headings = [actor_heading(setup, i, a[1], a[2], drone_pos) for (i, a) in enumerate(xy)]
    centers = [actor_center_uv(mesh, a[1], a[2], 0.0, drone_pos, yaw) for a in xy]
    infront = [c[3] > 0.05 for c in centers]
    tris = [infront[i] ? face_contributions(mesh, a[1], a[2], 0.0, h, drone_pos, yaw) : Tri[]
            for (i, (a, h)) in enumerate(zip(xy, headings))]
    ppa = [smooth_ppa_value(t) for t in tris]                                          # plain PPA
    appa = [[smooth_ppa_value(t, i, k) for k in 0:length(POWER_POINTS)] for (i, t) in enumerate(tris)]  # k = 0..4
    maxg = [[max_point_g(t, k) for k in eachindex(POWER_POINTS)] for t in tris]          # for the priority constraint
    infull = [infront[i] && whole_mesh_in_frame(mesh, a[1], a[2], 0.0, h, drone_pos, yaw)
              for (i, (a, h)) in enumerate(zip(xy, headings))]
    uv = [c[1:2] for c in centers]
    safe = all(hypot(drone_pos[1] - a[1], drone_pos[2] - a[2]) >= SAFETY_DIST for a in xy)
    return (ix=ix, iy=iy, drone_pos=drone_pos, yaw=yaw, headings=headings, ppa=ppa, appa=appa, maxg=maxg,
        uv=uv, safe=safe, infront=infront, infull=infull)
end

function evaluate_poses(mesh::TriMeshStruct, setup)
    poses = NamedTuple[]
    for (ix, dx) in enumerate(DRONE_XS), (iy, dy) in enumerate(DRONE_YS), dz in DRONE_ZS, off in YAW_OFFSETS
        yaw = midpoint_yaw(dx, dy, setup.xy) + off
        push!(poses, evaluate_pose(mesh, setup, [dx, dy, dz], yaw; ix=ix, iy=iy))
    end
    return poses
end

placement_fail(n::Int) = (objective=0.0, ok=false, assign=ntuple(_ -> 0, n), scores=ntuple(_ -> 0.0, n),
    errs=ntuple(_ -> 0.0, n))

# Scores one pose once an experiment's placement rule has produced the actors'
# assigned points `assign` (0 = none): applies the priority constraint and
# returns the view reward Σ PPA_i.
function placement_result(pose, assign)
    n = length(pose.uv)
    priority_ok(pose, assign[1]) || return placement_fail(n)
    scores = actor_ppas(pose, assign)
    errs = Tuple(point_distance(pose.uv[i], assign[i]) for i in 1:n)
    return (objective=sum(scores), ok=true, assign=assign, scores=scores, errs=errs)
end

# The pose with the largest view reward under an experiment's score function
# `score(pose)` (returns objective, ok, assign, scores, errs), plus a heatmap of
# the best reward per drone ground position (0 where no pose is feasible).
function best_over_poses(poses, score)
    grid = zeros(length(DRONE_YS), length(DRONE_XS))
    best = (objective=-Inf, pose=nothing, assign=(), scores=(), errs=())
    for pose in poses
        r = score(pose)
        r.ok || continue
        grid[pose.iy, pose.ix] = max(grid[pose.iy, pose.ix], r.objective)
        r.objective > best.objective && (best = (objective=r.objective, pose=pose, assign=r.assign, scores=r.scores, errs=r.errs))
    end
    return best, grid
end

# ── World View (3D) ───────────────────────────────────────────────────────────
function draw_quadcopter3d!(p, drone_pos::Vector{Float64}, drone_yaw::Float64; arm_length=0.3, prop_radius=0.15)
    x, y, z = drone_pos
    scatter!(p, [x], [y], [z], markersize=8, color=:darkred,
        markerstrokewidth=2, markerstrokecolor=:black, label="")
    for (i, arm_offset) in enumerate([π / 4, 3π / 4, 5π / 4, 7π / 4])
        angle = drone_yaw + arm_offset
        arm_x = x + arm_length * cos(angle)
        arm_y = y + arm_length * sin(angle)
        plot!(p, [x, arm_x], [y, arm_y], [z, z], color=:black, linewidth=3, label="")
        scatter!(p, [arm_x], [arm_y], [z], markersize=4, color=:gray,
            markerstrokewidth=1, markerstrokecolor=:black, label="")
        prop_θ = range(0, 2π; length=20)
        prop_x = arm_x .+ prop_radius .* cos.(prop_θ) .* cos(angle) .- prop_radius .* sin.(prop_θ) .* sin(angle)
        prop_y = arm_y .+ prop_radius .* cos.(prop_θ) .* sin(angle) .+ prop_radius .* sin.(prop_θ) .* cos(angle)
        plot!(p, prop_x, prop_y, fill(z + 0.05, length(prop_θ)),
            color=(i % 2 == 0 ? :red : :blue), linewidth=2, alpha=0.6, label="")
    end
    cam_x = x + 0.15 * cos(drone_yaw)
    cam_y = y + 0.15 * sin(drone_yaw)
    plot!(p, [x, cam_x], [y, cam_y], [z, z - 0.1], color=:lime, linewidth=3, label="")
    scatter!(p, [cam_x], [cam_y], [z - 0.1], markersize=3, color=:lime, marker=:square, label="")
end

function draw_actor_3d!(p, mesh::TriMeshStruct, x::Float64, y::Float64, z::Float64, heading::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, x, y, z, heading)
    for face in mesh.faces
        corners = [world_verts[idx] for idx in face.corner_indices]
        xs_f = [c[1] for c in corners]
        push!(xs_f, xs_f[1])
        ys_f = [c[2] for c in corners]
        push!(ys_f, ys_f[1])
        zs_f = [c[3] for c in corners]
        push!(zs_f, zs_f[1])
        plot!(p, xs_f, ys_f, zs_f, color=part_color(face.name), linewidth=2.0, label="")
    end
    arrow_len = 0.8
    plot!(p, [x, x + arrow_len * cos(heading)], [y, y + arrow_len * sin(heading)], [z, z],
        color=:green, linewidth=4, arrow=true, label="")
end

function fov_frustum_corners(drone_pos::Vector{Float64}, drone_yaw::Float64, depth::Float64)
    f = FOCAL_LENGTH
    corners = Vector{Float64}[]
    for (su, sv) in [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        s = sv * IMG.h * depth / f
        bx = depth * cos(TILT) - s * sin(TILT)
        by = su * IMG.w * depth / f
        dz = depth * sin(TILT) + s * cos(TILT)
        dx = bx * cos(drone_yaw) - by * sin(drone_yaw)
        dy = bx * sin(drone_yaw) + by * cos(drone_yaw)
        push!(corners, [drone_pos[1] + dx, drone_pos[2] + dy, drone_pos[3] + dz])
    end
    return corners
end

function draw_fov_frustum!(p, drone_pos::Vector{Float64}, drone_yaw::Float64; depth::Float64=2.0)
    corners = fov_frustum_corners(drone_pos, drone_yaw, depth)
    for c in corners
        plot!(p, [drone_pos[1], c[1]], [drone_pos[2], c[2]], [drone_pos[3], c[3]],
            color=:gray, linestyle=:dash, linewidth=1, alpha=0.6, label="")
    end
    xs_f = [c[1] for c in corners]
    push!(xs_f, xs_f[1])
    ys_f = [c[2] for c in corners]
    push!(ys_f, ys_f[1])
    zs_f = [c[3] for c in corners]
    push!(zs_f, zs_f[1])
    plot!(p, xs_f, ys_f, zs_f, color=:gray, linestyle=:dash, linewidth=1, alpha=0.6, label="")
end

function build_world_view(mesh::TriMeshStruct, actors::Vector{ActorPlacement},
    drone_pos::Vector{Float64}, drone_yaw::Float64)
    p_world = plot(
        xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="World View",
        legend=false, camera=(30, 45), aspect_ratio=:equal,
        xlims=extrema(DRONE_XS), ylims=extrema(DRONE_YS), zlims=(0.0, 5.0),
        background_color=:white
    )
    for a in actors
        draw_actor_3d!(p_world, mesh, a.x, a.y, 0.0, a.heading)
        plot!(p_world, [drone_pos[1], a.x], [drone_pos[2], a.y], [drone_pos[3], mesh.height / 2],
            linestyle=:dash, color=:gray, linewidth=1, alpha=0.6, label="")
    end
    draw_quadcopter3d!(p_world, drone_pos, drone_yaw)
    draw_fov_frustum!(p_world, drone_pos, drone_yaw)
    return p_world
end

# ── FPV Camera ────────────────────────────────────────────────────────────────
function draw_ground_grid!(p, drone_pos::Vector{Float64}, drone_yaw::Float64; grid_range=-8:2:8)
    for gx in grid_range
        pa = project_or_nothing([Float64(gx), Float64(first(grid_range)), 0.0], drone_pos, drone_yaw)
        pb = project_or_nothing([Float64(gx), Float64(last(grid_range)), 0.0], drone_pos, drone_yaw)
        (pa === nothing || pb === nothing) || plot!(p, [pa[1], pb[1]], [pa[2], pb[2]], color=:gray40, linewidth=0.8, alpha=0.5, label="")
    end
    for gy in grid_range
        pa = project_or_nothing([Float64(first(grid_range)), Float64(gy), 0.0], drone_pos, drone_yaw)
        pb = project_or_nothing([Float64(last(grid_range)), Float64(gy), 0.0], drone_pos, drone_yaw)
        (pa === nothing || pb === nothing) || plot!(p, [pa[1], pb[1]], [pa[2], pb[2]], color=:gray40, linewidth=0.8, alpha=0.5, label="")
    end
end

function draw_fpv_actor!(p, mesh::TriMeshStruct, a::ActorPlacement, drone_pos::Vector{Float64}, drone_yaw::Float64)
    world_verts = ActorMeshTriangulated.actor_world_vertices(mesh, a.x, a.y, 0.0, a.heading)
    for face in mesh.faces
        proj = [project_or_nothing(world_verts[idx], drone_pos, drone_yaw) for idx in face.corner_indices]
        any(pt -> pt === nothing, proj) && continue
        plot!(p, Shape([pt[1] for pt in proj], [pt[2] for pt in proj]), fillalpha=0.65,
            fillcolor=part_color(face.name), linecolor=:black, linewidth=0.5, label="")
    end
end

# Thirds grid and power points; the points assigned to an actor are gold stars,
# each actor's center is marked and labeled A1, A2, ...
function draw_thirds_and_centers!(p, uv, assign)
    W, H = IMG.w, IMG.h
    for uu in (2W / 3, 4W / 3)
        plot!(p, [uu, uu], [0, 2H], color=:gray50, linestyle=:dash, linewidth=1, label="")
    end
    for vv in (2H / 3, 4H / 3)
        plot!(p, [0, 2W], [vv, vv], color=:gray50, linestyle=:dash, linewidth=1, label="")
    end
    for (k, pp) in enumerate(POWER_POINTS)
        used = k in assign
        scatter!(p, [pp.u], [pp.v], shape=(used ? :star5 : :circle), ms=(used ? 9 : 5),
            mc=(used ? :gold : :white), msc=:black, label="")
    end
    for (i, (u, v)) in enumerate(uv)
        scatter!(p, [u], [v], shape=:xcross, ms=6, mc=:black, label="")
        annotate!(p, u, v + 0.07, text("A$i", :black, :center, 8))   # label below the center (+v down)
    end
end

function build_fpv_view(mesh::TriMeshStruct, actors::Vector{ActorPlacement},
    drone_pos::Vector{Float64}, drone_yaw::Float64, uv, assign)
    p_fpv = plot(title="FPV Camera (16:9)", legend=false,
        xlims=(0, 2IMG.w), ylims=(0, 2IMG.h), yflip=true, aspect_ratio=:equal,   # origin top-left, +v down
        grid=false, ticks=false, framestyle=:box)
    draw_ground_grid!(p_fpv, drone_pos, drone_yaw)
    for a in actors
        draw_fpv_actor!(p_fpv, mesh, a, drone_pos, drone_yaw)
    end
    draw_thirds_and_centers!(p_fpv, uv, assign)
    tilt_deg = round(Int, TILT * 180 / π)
    annotate!(p_fpv, 0.02, 2IMG.h - 0.06, text("ALT  $(round(drone_pos[3], digits=1))m", :left, :lime, 7))
    annotate!(p_fpv, 0.02, 2IMG.h - 0.12, text("TILT $(tilt_deg)°", :left, :dodgerblue, 7))
    annotate!(p_fpv, 2IMG.w - 0.02, 0.04, text("● REC", :right, :red, 8))
    return p_fpv
end

# ── Actor heatmap: drone fixed, actors moved over the ground ──────────────────
# Actor i's PPA (composition term inside, measured to its assigned power point
# k, or the nearest one when k = 0) at every actor ground position, with the
# drone fixed and the actor at its setup heading. 0 where the actor's center is
# behind the camera, and for non-priority actors also where they would occlude
# actor 1's spot `block_k`.
function actor_position_grid(mesh::TriMeshStruct, i::Int, k::Int, setup, drone_pos::Vector{Float64}, yaw::Float64;
    plain_ppa::Bool=false, block_k::Int=0)
    grid = zeros(length(ACTOR_GRID_YS), length(ACTOR_GRID_XS))
    for (jx, x) in enumerate(ACTOR_GRID_XS), (jy, y) in enumerate(ACTOR_GRID_YS)
        _, _, depth = actor_center_uv(mesh, x, y, 0.0, drone_pos, yaw)
        depth > 0.05 || continue
        tris = face_contributions(mesh, x, y, 0.0, actor_heading(setup, i, x, y, drone_pos), drone_pos, yaw)
        blocks = i > 1 && block_k > 0 && 1 - max_point_g(tris, block_k) < PRIORITY_AWAY_MIN
        grid[jy, jx] = blocks ? 0.0 : plain_ppa ? smooth_ppa_value(tris) : smooth_ppa_value(tris, i, k)
    end
    return grid
end

# All actors' PPA maps on one heatmap (each cell shows the largest).
# plain_ppa = true: no composition term and no priority blocking (plain PPA only).
function actor_positions_panel(mesh::TriMeshStruct, setup, pose, actors, assign; plain_ppa::Bool=false)
    n = length(actors)
    spot1 = priority_point(pose.uv[1], assign[1])   # actor 1's spot, which the others must not occlude
    grids = [actor_position_grid(mesh, i, assign[i], setup, pose.drone_pos, pose.yaw;
        plain_ppa=plain_ppa, block_k=plain_ppa ? 0 : spot1) for i in 1:n]
    combined = reduce((a, b) -> max.(a, b), grids)
    hi = max(maximum(combined), 1e-9)
    title = plain_ppa ? "Actor View Reward vs actor position \nplain View Reward, no composition" :
            n == 1 ? "Actor View Reward vs actor position" :
            "Actor View Reward vs actor position "
    p = heatmap(ACTOR_GRID_XS, ACTOR_GRID_YS, combined, c=:jet, clims=(0.0, hi), titlefontsize=10, title=title,
        xlabel="Actor X (m)", ylabel="Actor Y (m)", aspect_ratio=:equal,
        xlims=extrema(ACTOR_GRID_XS), ylims=extrema(ACTOR_GRID_YS), framestyle=:box, left_margin=12Plots.mm)
    # Where an actor has to stand for its center to land on each power point;
    # the points assigned to an actor are gold stars.
    for (kk, pp) in enumerate(POWER_POINTS)
        g = uv_to_ground(pp.u, pp.v, mesh.height / 2, pose.drone_pos, pose.yaw)
        (g === nothing || !(first(ACTOR_GRID_XS) <= g[1] <= last(ACTOR_GRID_XS)) ||
         !(first(ACTOR_GRID_YS) <= g[2] <= last(ACTOR_GRID_YS))) && continue
        used = kk in assign
        scatter!(p, [g[1]], [g[2]], shape=(used ? :star5 : :diamond), ms=(used ? 9 : 5),
            mc=(used ? :gold : :white), msc=:black, label="")
        annotate!(p, g[1] + 0.4, g[2] + 0.4, text(POINT_ABBREV[pp.name], :white, :left, 7))
    end
    scatter!(p, [pose.drone_pos[1]], [pose.drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="")
    quiver!(p, [pose.drone_pos[1]], [pose.drone_pos[2]], quiver=([0.9 * cos(pose.yaw)], [0.9 * sin(pose.yaw)]), color=:white, lw=2.5)
    for (j, a) in enumerate(actors)
        scatter!(p, [a.x], [a.y], shape=:circle, ms=7, mc=:white, msc=:black, label="")
        annotate!(p, a.x + 0.5, a.y - 0.6, text("A$j", :white, :left, 8))
    end
    return p
end

# Display only: keeps the connected region of feasible cells (value > 0,
# 8-connected) that contains cell (iy, ix), the best pose; every other cell is
# set to 0. The search and the best pose are not affected.
function main_region_only(grid, iy::Int, ix::Int)
    out = zeros(size(grid))
    grid[iy, ix] > 0 || return out
    stack = [(iy, ix)]
    seen = falses(size(grid))
    seen[iy, ix] = true
    while !isempty(stack)
        (r, c) = pop!(stack)
        out[r, c] = grid[r, c]
        for dr in -1:1, dc in -1:1
            rr, cc = r + dr, c + dc
            (1 <= rr <= size(grid, 1) && 1 <= cc <= size(grid, 2)) || continue
            (!seen[rr, cc] && grid[rr, cc] > 0) || continue
            seen[rr, cc] = true
            push!(stack, (rr, cc))
        end
    end
    return out
end

# ── One experiment figure ─────────────────────────────────────────────────────
# `star` lists the power points marked with a gold star; `parts` are the
# per-actor summaries shown under the title.
function render_experiment(mesh::TriMeshStruct, setup, pose, star, title::String, heat_title::String,
    grid, clims, parts, out_file::String; plain_ppa::Bool=false, main_region::Bool=false)
    actors = [ActorPlacement(a[1], a[2], h) for (a, h) in zip(setup.xy, pose.headings)]
    main_region && (grid = main_region_only(grid, pose.iy, pose.ix))   # display only: hide isolated feasible cells
    p_world = build_world_view(mesh, actors, pose.drone_pos, pose.yaw)
    p_fpv = build_fpv_view(mesh, actors, pose.drone_pos, pose.yaw, pose.uv, star)

    p_heat = heatmap(DRONE_XS, DRONE_YS, grid, c=:jet, clims=clims, title=heat_title, titlefontsize=10,
        xlabel="Drone X (m)", ylabel="Drone Y (m)", aspect_ratio=:equal,
        xlims=extrema(DRONE_XS), ylims=extrema(DRONE_YS), framestyle=:box, left_margin=12Plots.mm)
    scatter!(p_heat, [pose.drone_pos[1]], [pose.drone_pos[2]], shape=:rect, ms=8, mc=:white, msc=:black, label="")
    quiver!(p_heat, [pose.drone_pos[1]], [pose.drone_pos[2]], quiver=([0.9 * cos(pose.yaw)], [0.9 * sin(pose.yaw)]), color=:white, lw=2.5)
    for (i, a) in enumerate(actors)
        scatter!(p_heat, [a.x], [a.y], shape=:circle, ms=7, mc=:white, msc=:black, label="")
        annotate!(p_heat, a.x + 0.6, a.y + 0.6, text("A$i", :black, :left, 8))
    end

    p_actors = actor_positions_panel(mesh, setup, pose, actors, star; plain_ppa=plain_ppa)

    # The per-actor summary (`parts`) is printed to the console only, not drawn in the image.
    p_combined = plot(p_world, p_fpv, p_heat, p_actors, layout=(1, 4),
        size=(2400, 660), plot_title=title, plot_titlefontsize=11, plot_titlevspan=0.1)

    mkpath(dirname(out_file))
    savefig(p_combined, out_file)
    d = pose.drone_pos
    println("✓ Saved → $out_file")
    println("    drone=($(round(d[1], digits=2)), $(round(d[2], digits=2)), $(d[3]))  yaw=$(round(rad2deg(pose.yaw), digits=1))°  " *
            join(parts, "  "))
end

# Per-actor summary line (printed to the console only, not drawn in the figures).
actor_summary(i, where_, pose, score) =
    "A$i: $where_, PPA=$(round(score, digits=4)) (plain $(round(pose.ppa[i], digits=4)))"

# Summary of every actor's placement and PPA for the best pose of an experiment.
function placement_parts(best)
    map(eachindex(best.assign)) do i
        k = best.assign[i]
        where_ = k == 0 ? "off (nearest point $(round(best.errs[i], digits=3)))" :
                 "$(POWER_POINTS[k].name), d=$(round(best.errs[i], digits=3))"
        actor_summary(i, where_, best.pose, best.scores[i])
    end
end
