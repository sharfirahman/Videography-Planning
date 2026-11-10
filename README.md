# Composition-Aware Drone Filming

Plans camera poses for a filming drone so that one or more actors are framed
according to photographic composition rules (rule of thirds, centering, close
shots), while still capturing as much useful pixel area of each actor as
possible.

The active research code lives in [`src/CompositionAwareFilming/`](src/CompositionAwareFilming/).



## Actor Model

Each actor is a triangulated human mesh
([`simple_human_rotated_xaxis.obj`](src/CompositionAwareFilming/simple_human_rotated_xaxis.obj)),
with faces grouped into head, body, feet and top. The camera is a pinhole
camera tilted downward, with a 16:9 image.

The quality of a drone pose is its **view reward**, the sum of the Pixels Per
Area (PPA) of every actor:

```
reward = Σ_i PPA_i,    PPA_i = Σ_faces  ω · γ · D · g
```

| Term | Meaning |
|------|---------|
| `ω`  | Heading weight: faces the actor is turned toward count more. |
| `γ`  | Visibility gate: the face points at the camera and all three vertices are in front of the camera and inside the image. |
| `D`  | Pixel density, `sqrt(A_face · A_image)`: real face area times its projected area. |
| `g`  | Composition term: a Gaussian centred on the actor's preferred rule-of-thirds power point, so the actor is pulled toward that point in the image. |

Without `g` this is "plain PPA". The repo has two PPA models: the **original**
one with a hard visibility test, and a **smooth** one used in the current
experiments.

A pose is rejected (scores 0) unless it meets these constraints:

- **Safety:** the drone stays at least 1.5 m from every actor horizontally.
- **Framing:** the actors are fully inside the image (some experiments relax this).
- **Placement:** an experiment-specific rule on where each actor's centre lands
  relative to the power points.
- **Priority:** with two actors, actor 2's faces must stay away from actor 1's
  power point, so actor 2 never takes actor 1's spot in the frame.

The best pose is found by exhaustive search over drone position (a 16 m × 16 m
grid), altitude (1.5–4 m) and yaw (±30° around the direction to the actors).

## Repository layout

```
src/CompositionAwareFilming/     Active research code
  ViewPlanningCore.jl              Camera model, view reward, constraints, pose search, figures
  ActorMeshTriangulated.jl         Loads the OBJ actor mesh into weighted triangles
  Compositional_preferences.jl     Composition scores (rule of thirds, centre, close shot)
  Single_Actor_Experiments_Compositional_Preferences.jl   1 actor, 1 drone
  Multi_Actor_experiments_Compositional_Preferences.jl    2 actors, 1 drone
  Single_Actor_experiments.jl, Multi_Actor_experiments.jl Earlier framing scenarios
  StaticPPA_*.jl, OriginalPPAvssmooth.jl, Compare_PPA_*.jl Original vs smooth PPA studies
  drone_experiments/               All generated figures and GIFs
src/MPC_archive/                 Older MPC experiments
```

## Setup

The Julia environment in `env/` was resolved with **Julia 1.12**.

```bash
git clone git@github.com:sharfirahman/Videography-Planning.git
cd Videography-Planning
julia --project=env -e 'using Pkg; Pkg.instantiate()'
```

## Running the experiments

Run every script **from the repository root**: output paths are relative to it,
and figures are written to `src/CompositionAwareFilming/drone_experiments/`.

Composition experiments (the current main results):

```bash
# One actor placed on each of the four rule-of-thirds power points
julia --project=env src/CompositionAwareFilming/Single_Actor_Experiments_Compositional_Preferences.jl

# Two actors: both on power points, one on and one near, neither on, and
# experiments with fixed actor headings
julia --project=env src/CompositionAwareFilming/Multi_Actor_experiments_Compositional_Preferences.jl
```

Each experiment saves one figure with three panels: a 3D world view with the
drone and its field of view, the drone's camera image with the power points,
and a heatmap of the reward as the actors move over the ground.

Both scripts can also be `include`d from a Julia REPL. When you edit
`ViewPlanningCore.jl`, it is reloaded on the next include, so you don't need to
restart Julia.

## License

BSD 3-Clause. See [`LICENSE`](LICENSE).
