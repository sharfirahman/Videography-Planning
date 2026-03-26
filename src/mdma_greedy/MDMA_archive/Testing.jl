using GLMakie

fig = Figure(resolution = (1400, 700))

# World view
ax_world = Axis3(fig[1, 1], title="World View")

# Camera view
ax_cam = Axis(fig[1, 2], title="Drone FPV", 
              limits=(-2, 2, -2, 2))

# Observables for animation
drone_pos_obs = Observable(Point3f(0, 0, 1))
human_pos_obs = Observable(Point3f(2, 0, 0))

# World view visualization
scatter!(ax_world, drone_pos_obs, color=:blue, markersize=20)
scatter!(ax_world, human_pos_obs, color=:red, markersize=20)

# Camera view (projected)
human_in_cam = lift(drone_pos_obs, human_pos_obs) do d_pos, h_pos
    # Simple projection (relative position from drone perspective)
    relative = h_pos - d_pos
    return Point2f(relative[2], relative[3])  # y-z projection
end

scatter!(ax_cam, human_in_cam, color=:red, markersize=30)

display(fig)