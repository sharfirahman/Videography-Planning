# Compositional_preferences.jl
# Helper functions scoring different photographic/cinematographic composition
# preferences in normalized image space (u, v roughly in [-1, 1], matching the
# camera projection convention used elsewhere: u = focal_length*cy/cx,
# v = focal_length*cz/cx). Each function returns a score in (0, 1], highest
# when the composition matches that preference.

module CompositionalPreferences

export rule_of_thirds_score, center_placement_score, close_shot_score, balance_score

# ── 1. Rule of Thirds ──────────────────────────────────────────────────────
# Reward the subject sitting near one of the four rule-of-thirds intersection
# points instead of dead center. `k` controls how sharply the score falls off
# with distance from the nearest intersection point.
const THIRDS_POINTS = [(-1/3, -1/3), (-1/3, 1/3), (1/3, -1/3), (1/3, 1/3)]

function rule_of_thirds_score(u::Float64, v::Float64; k::Float64 = 4.0)
    d2 = minimum((u - pu)^2 + (v - pv)^2 for (pu, pv) in THIRDS_POINTS)
    return exp(-k * d2)
end

# ── 2. Center Placement ────────────────────────────────────────────────────
# Reward the subject sitting at the dead center of the frame.
function center_placement_score(u::Float64, v::Float64; k::Float64 = 4.0)
    return exp(-k * (u^2 + v^2))
end

# ── 3. Close Shot (face-filling) ───────────────────────────────────────────
# Reward the actor's face/head occupying a large fraction of the frame.
# `head_uv` are the projected (u, v) points of the head's vertices/corners;
# the score is the fraction of the full frame area ([-1,1]x[-1,1], area 4)
# covered by the head's axis-aligned bounding box. Points are clamped to the
# visible frame first, so a head that's fully or partly off-screen (e.g. the
# actor is too close and has drifted out of frame) correctly scores low
# instead of appearing to fill the frame via an off-screen bounding box.
function close_shot_score(head_uv::Vector{Tuple{Float64,Float64}})
    isempty(head_uv) && return 0.0
    us = clamp.(first.(head_uv), -1.0, 1.0)
    vs = clamp.(last.(head_uv), -1.0, 1.0)
    width  = maximum(us) - minimum(us)
    height = maximum(vs) - minimum(vs)
    return (width * height) / 4.0
end

# ── 4. Balance (multi-subject framing) ─────────────────────────────────────
# Reward multiple subjects being visually balanced around the frame center
# rather than clustered to one side, using the classic composition idea that
# a scene "balances" when its subjects' weighted centroid sits near the
# center (like a mobile balancing around a pivot). `positions` are each
# subject's (u, v) frame position; `weights` lets a subject count for more
# (e.g. a closer/larger actor carries more visual weight) — defaults to equal.
function balance_score(positions::Vector{Tuple{Float64,Float64}}, weights::Vector{Float64} = fill(1.0, length(positions)); k::Float64 = 4.0)
    isempty(positions) && return 0.0
    total_weight = sum(weights)
    total_weight <= 0.0 && return 0.0
    cu = sum(w * u for (w, (u, _)) in zip(weights, positions)) / total_weight
    cv = sum(w * v for (w, (_, v)) in zip(weights, positions)) / total_weight
    return exp(-k * (cu^2 + cv^2))
end

end # module
