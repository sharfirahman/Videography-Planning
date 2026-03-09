using CSV, DataFrames, Plots
ENV["GKSwstype"] = "100"
df = CSV.read("figure8_drone_trajectory.csv", DataFrame)

# Plot the velocities over time
plot(df.step, [df.vx df.vy df.vz],
    label=["vx (Forward/Backward)" "vy (Left/Right)" "vz (Up/Down)"],
    xlabel="Time Step",
    ylabel="Velocity (m/s)",
    title="Drone Velocities (Figure 8 Trajectory)",
    linewidth=2,
    legend=:outertopright,
    size=(800, 400))

# Add a zero-line for reference
hline!([0], color=:black, linestyle=:dash, label="")

# Save the plot
savefig("velocity_plot.png")
println("Saved plot to velocity_plot.png")
