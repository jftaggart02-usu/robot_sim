# robot_sim

A modular mobile-robot simulation framework built in Python. The simulator uses an RRT* motion planner to find a collision-free path through polygon obstacles, converts that path into a time-indexed trajectory, and tracks it with a PID controller driving a unicycle-model vehicle — all rendered in real time with matplotlib.

---

## Table of Contents

- [Getting Started](#getting-started)
  - [Clone the Repository](#1-clone-the-repository)
  - [Create a Virtual Environment](#2-create-a-virtual-environment)
  - [Install Dependencies](#3-install-dependencies)
  - [Run the Simulation](#4-run-the-simulation)
- [Project Structure](#project-structure)
- [Public API](#public-api)
  - [robot\_sim.types](#robot_simtypes)
  - [robot\_sim.obstacles](#robot_simobstacles)
  - [robot\_sim.planner](#robot_simplanner)
  - [robot\_sim.trajectory](#robot_simtrajectory)
  - [robot\_sim.controller](#robot_simcontroller)
  - [robot\_sim.dynamics](#robot_simdynamics)
  - [robot\_sim.visualizer](#robot_simvisualizer)
- [Running the Tests](#running-the-tests)

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jftaggart02-usu/robot_sim.git
cd robot_sim
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project requires Python 3.9+ and the following packages (pinned in `requirements.txt`):

| Package | Purpose |
|---------|---------|
| `numpy>=1.26` | Numerical arrays used by the visualizer |
| `matplotlib>=3.8` | Real-time simulation display |
| `shapely>=2.0` | Polygon intersection / collision detection |

### 4. Run the Simulation

**Single robot (default):**

```bash
python main.py
```

The script:

1. Builds a `SimConfig` with a start state, goal, and several polygon obstacles.
2. Runs the RRT* planner to find a collision-free path.
3. Converts the path to a time-indexed trajectory.
4. Enters the main simulation loop — sample desired state → PID control → unicycle dynamics → display update.
5. Saves a final screenshot to `sim_result.png` in the working directory.

**Multiple robots simultaneously:**

```bash
python main.py --multi-robot
```

Each robot gets its own RRT* plan, trajectory, and PID controller instance that are all stepped together at every simulation tick. Results are saved to `sim_multi_result.png` and (by default) `sim_multi_animation.gif`.

**Common flags:**

| Flag | Description |
|------|-------------|
| `--multi-robot` | Run the 3-robot demo instead of the single-robot demo |
| `--interactive` | Show a live matplotlib window during simulation |
| `--save-path FILE` | Override the output image path |
| `--no-animate` | Skip saving the GIF animation |

---

## Project Structure

```
robot_sim/
├── main.py               # Runnable end-to-end example
├── requirements.txt      # Python dependencies
├── robot_sim/            # Library package
│   ├── __init__.py
│   ├── types.py          # Shared dataclasses (pure data)
│   ├── obstacles.py      # Polygon collision-detection helpers
│   ├── planner.py        # RRT* motion planner
│   ├── trajectory.py     # Time-indexed trajectory generation
│   ├── controller.py     # PID controller
│   ├── dynamics.py       # Unicycle-model forward dynamics
│   └── visualizer.py     # matplotlib real-time display
└── tests/
    └── test_sim.py       # pytest unit / integration tests
```

---

## Public API

All public symbols live inside the `robot_sim` package. Import them from their respective modules as shown below.

---

### `robot_sim.types`

Shared dataclasses — carry data only, no logic.

#### `PolygonObstacle`

```python
@dataclass
class PolygonObstacle:
    vertices: List[Tuple[float, float]]
```

A filled 2-D polygon defined by an ordered list of `(x, y)` vertices.

---

#### `VehicleState`

```python
@dataclass
class VehicleState:
    x: float      # position x  (m)
    y: float      # position y  (m)
    theta: float  # heading     (rad, measured from +x axis)
    v: float      # speed       (m/s)
```

Full state of the unicycle-model vehicle.

---

#### `ControlInput`

```python
@dataclass
class ControlInput:
    a: float      # linear acceleration  (m/s²)
    omega: float  # angular velocity     (rad/s)
```

Control input applied to the dynamics model.

---

#### `Waypoint` / `Path`

```python
@dataclass
class Waypoint:
    x: float
    y: float

@dataclass
class Path:
    waypoints: List[Waypoint]
```

An ordered sequence of 2-D position waypoints produced by the motion planner.

---

#### `TrajectoryPoint` / `Trajectory`

```python
@dataclass
class TrajectoryPoint:
    t: float      # time since trajectory start  (s)
    x: float      # desired position x            (m)
    y: float      # desired position y            (m)
    theta: float  # desired heading               (rad)
    v: float      # desired speed                 (m/s)

@dataclass
class Trajectory:
    points: List[TrajectoryPoint]
```

A time-indexed sequence of desired vehicle states.

---

#### `PIDGains` / `PIDControllerConfig` / `PIDState` / `PIDControllerState`

```python
@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float

@dataclass
class PIDControllerConfig:
    heading_gains: PIDGains   # controls omega (angular velocity)
    speed_gains: PIDGains     # controls a     (linear acceleration)

@dataclass
class PIDState:
    integral: float = 0.0
    prev_error: float = 0.0

@dataclass
class PIDControllerState:
    heading: PIDState = field(default_factory=PIDState)
    speed: PIDState = field(default_factory=PIDState)
```

PID gains and mutable integrator / derivative state for the two controller channels (heading and speed).

---

#### `SimConfig`

```python
@dataclass
class SimConfig:
    initial_state: VehicleState
    goal: Tuple[float, float]                       # (x, y) goal position
    obstacles: List[PolygonObstacle]
    bounds: Tuple[float, float, float, float]        # (x_min, x_max, y_min, y_max)
    dt: float = 0.05                                 # simulation timestep  (s)
    max_time: float = 60.0                           # maximum simulation duration  (s)
    goal_tolerance: float = 0.3                      # distance to consider goal reached  (m)
    cruise_speed: float = 1.5                        # nominal travel speed along trajectory (m/s)
    max_speed: float = 3.0                           # speed saturation limit  (m/s)
    max_accel: float = 2.0                           # acceleration saturation limit (m/s²)
    max_omega: float = 1.5                           # angular-velocity saturation limit (rad/s)
    rrt_max_iter: int = 3000
    rrt_step_size: float = 0.5
    rrt_goal_bias: float = 0.1                       # probability of sampling the goal directly
    rrt_neighbor_radius: float = 1.5                 # rewire neighborhood radius  (m)
    pid: PIDControllerConfig = ...                   # default: kp=2.5/1.5, ki=0/0.1, kd=0.3/0.05
```

Top-level configuration for a **single-robot** simulation.

---

#### `RobotConfig`

```python
@dataclass
class RobotConfig:
    initial_state: VehicleState
    goal: Tuple[float, float]          # (x, y) goal position
    label: str = ""                    # display label (e.g. "Robot A")
    color: str = "red"                 # marker/trail color for the visualizer
    goal_color: Optional[str] = None   # goal-star color (defaults to color)
    cruise_speed: float = 1.5          # nominal travel speed  (m/s)
    max_speed: float = 3.0             # speed saturation  (m/s)
    max_accel: float = 2.0             # acceleration saturation  (m/s²)
    max_omega: float = 1.5             # angular-velocity saturation  (rad/s)
    goal_tolerance: float = 0.3        # distance to consider goal reached  (m)
    rrt_max_iter: int = 3000
    rrt_step_size: float = 0.5
    rrt_goal_bias: float = 0.1
    rrt_neighbor_radius: float = 1.5
    rrt_seed: Optional[int] = None     # reproducible RRT* seed (None = non-deterministic)
    pid: PIDControllerConfig = ...     # per-robot PID gains
```

Per-robot configuration for a **multi-robot** simulation. Each `RobotConfig` holds its own start state, goal, motion parameters, PID gains, and visual style.

---

#### `MultiRobotSimConfig`

```python
@dataclass
class MultiRobotSimConfig:
    robots: List[RobotConfig]
    obstacles: List[PolygonObstacle]
    bounds: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    dt: float = 0.05                           # simulation timestep  (s)
    max_time: float = 60.0                     # maximum simulation duration  (s)
```

Top-level configuration for a **multi-robot** simulation. Shared environment settings (obstacles, workspace bounds, timestep) live here; per-robot settings live inside each `RobotConfig`.

---

### `robot_sim.obstacles`

Pure functions for polygon obstacle collision detection, backed by [Shapely](https://shapely.readthedocs.io/).

#### `point_in_obstacle`

```python
def point_in_obstacle(x: float, y: float, obstacle: PolygonObstacle) -> bool:
```

Return `True` if the 2-D point `(x, y)` lies inside `obstacle`.

#### `point_in_any_obstacle`

```python
def point_in_any_obstacle(x: float, y: float, obstacles: List[PolygonObstacle]) -> bool:
```

Return `True` if `(x, y)` lies inside any obstacle in the list.

#### `segment_collides_with_obstacle`

```python
def segment_collides_with_obstacle(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    obstacle: PolygonObstacle,
) -> bool:
```

Return `True` if the line segment `p1 → p2` intersects `obstacle`.

#### `segment_collides_with_any`

```python
def segment_collides_with_any(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    obstacles: List[PolygonObstacle],
) -> bool:
```

Return `True` if the segment `p1 → p2` intersects any obstacle in the list.

#### `to_shapely`

```python
def to_shapely(obstacle: PolygonObstacle) -> shapely.geometry.Polygon:
```

Convert a `PolygonObstacle` to a Shapely `Polygon` for custom geometry queries.

---

### `robot_sim.planner`

RRT* (Rapidly-exploring Random Tree Star) motion planner.

#### `plan`

```python
def plan(
    initial_state: VehicleState,
    goal: Tuple[float, float],
    obstacles: List[PolygonObstacle],
    bounds: Tuple[float, float, float, float],
    max_iter: int = 3000,
    step_size: float = 0.5,
    goal_bias: float = 0.1,
    neighbor_radius: float = 1.5,
    seed: Optional[int] = None,
) -> Optional[Path]:
```

Run RRT* and return a collision-free `Path`, or `None` if planning fails within `max_iter` iterations.

| Parameter | Description |
|-----------|-------------|
| `initial_state` | Start state of the vehicle (x, y used as root). |
| `goal` | `(x, y)` goal position. |
| `obstacles` | List of polygon obstacles the path must avoid. |
| `bounds` | `(x_min, x_max, y_min, y_max)` workspace bounds for random sampling. |
| `max_iter` | Maximum number of RRT* iterations. |
| `step_size` | Maximum branch length at each extension step (m). |
| `goal_bias` | Probability of sampling the goal position directly (0–1). |
| `neighbor_radius` | Radius for the rewire neighborhood search (m). |
| `seed` | Optional integer random seed for reproducibility. |

**Example**

```python
from robot_sim.planner import plan
from robot_sim.types import VehicleState, PolygonObstacle

start = VehicleState(x=0.5, y=0.5, theta=0.0, v=0.0)
goal  = (9.5, 9.5)
wall  = PolygonObstacle(vertices=[(4.5, 0.0), (5.5, 0.0), (5.5, 10.0), (4.5, 10.0)])

path = plan(start, goal, obstacles=[wall], bounds=(0.0, 10.0, 0.0, 10.0), seed=42)
if path:
    print(f"Found path with {len(path.waypoints)} waypoints")
```

---

### `robot_sim.trajectory`

Time-indexed trajectory generation from a planned path.

#### `build_trajectory`

```python
def build_trajectory(path: Path, cruise_speed: float = 1.5) -> Trajectory:
```

Convert a `Path` into a time-indexed `Trajectory`.

- Timestamps are assigned based on a constant `cruise_speed`.
- The first point has speed `0`; the last point also has speed `0` (goal stop).
- Heading at each waypoint is taken from the direction of the outgoing segment.

#### `sample_trajectory`

```python
def sample_trajectory(trajectory: Trajectory, t: float) -> TrajectoryPoint:
```

Return the desired state at time `t` by linearly interpolating between the two surrounding `TrajectoryPoint`s. Heading is wrap-interpolated. Returns the first/last point if `t` is before/after the trajectory bounds.

**Example**

```python
from robot_sim.trajectory import build_trajectory, sample_trajectory

trajectory = build_trajectory(path, cruise_speed=1.5)
desired = sample_trajectory(trajectory, t=3.0)
print(desired.x, desired.y, desired.theta, desired.v)
```

---

### `robot_sim.controller`

PID controller for unicycle-model vehicles (pure functions).

#### `compute_control`

```python
def compute_control(
    desired: TrajectoryPoint,
    current: VehicleState,
    pid_config: PIDControllerConfig,
    pid_state: PIDControllerState,
    dt: float,
    max_omega: float = 1.5,
    max_accel: float = 2.0,
) -> Tuple[ControlInput, PIDControllerState]:
```

Compute a `ControlInput` from the error between `desired` and `current` state.

Two independent PID channels are used:

- **Heading channel** — steers toward the desired position (or desired heading when already close); outputs `omega` (rad/s).
- **Speed channel** — tracks the desired speed; outputs `a` (m/s²).

Both outputs are saturated to `[−max_omega, max_omega]` and `[−max_accel, max_accel]` respectively, with anti-windup on the integrators.

The caller is responsible for threading the returned `PIDControllerState` forward through time.

**Example**

```python
from robot_sim.controller import compute_control
from robot_sim.types import PIDControllerState

pid_state = PIDControllerState()
control, pid_state = compute_control(desired, current, pid_config, pid_state, dt=0.05)
```

---

### `robot_sim.dynamics`

Unicycle-model forward dynamics (pure functions).

#### `step`

```python
def step(
    state: VehicleState,
    control: ControlInput,
    dt: float,
    max_speed: float = 3.0,
    max_accel: float = 2.0,
    max_omega: float = 1.5,
) -> VehicleState:
```

Advance the vehicle state by one timestep using forward Euler integration.

The continuous equations of motion are:

```
ẋ = v · cos(θ)
ẏ = v · sin(θ)
θ̇ = ω
v̇ = a
```

Inputs `a` and `ω` are saturated before integration; speed is clamped to `[−max_speed, max_speed]`.

**Example**

```python
from robot_sim.dynamics import step as dynamics_step
from robot_sim.types import VehicleState, ControlInput

state   = VehicleState(x=0.0, y=0.0, theta=0.0, v=1.0)
control = ControlInput(a=0.5, omega=0.1)
state   = dynamics_step(state, control, dt=0.05)
```

---

### `robot_sim.visualizer`

matplotlib-based real-time simulation display (pure functions).

The display renders:
- Polygon obstacles (filled dark grey)
- RRT* path (dashed blue)
- Time-indexed trajectory (solid cyan)
- Current desired state (green circle + heading arrow)
- Current vehicle state (red triangle + heading arrow)
- Vehicle trail (salmon)
- Goal marker (gold star)

#### `init_display`

```python
def init_display(
    config: SimConfig,
    path: Path,
    trajectory: Trajectory,
    interactive: bool = True,
) -> DisplayState:
```

Create and return a `DisplayState` with all static elements drawn. Set `interactive=False` for headless / batch rendering.

#### `update_display`

```python
def update_display(
    ds: DisplayState,
    desired: TrajectoryPoint,
    vehicle: VehicleState,
    pause: float = 0.001,
    interactive: bool = True,
) -> None:
```

Redraw the dynamic elements (desired state marker, vehicle marker, heading arrows, trail) for the current simulation step.

#### `save_display`

```python
def save_display(ds: DisplayState, filepath: str) -> None:
```

Save the current figure to `filepath`. Supports any format matplotlib accepts (PNG, PDF, SVG, …).

#### `close_display`

```python
def close_display(ds: DisplayState) -> None:
```

Close the matplotlib figure and release resources.

**Example**

```python
import matplotlib
matplotlib.use("Agg")

from robot_sim.visualizer import init_display, update_display, save_display, close_display

ds = init_display(config, path, trajectory, interactive=False)
update_display(ds, desired, state, interactive=False)
save_display(ds, "result.png")
close_display(ds)
```

#### Multi-robot display functions

For multi-robot scenarios, use the parallel set of functions that accept `MultiRobotSimConfig` and per-robot lists:

```python
def init_multi_display(
    config: MultiRobotSimConfig,
    paths: List[Path],
    trajectories: List[Trajectory],
    interactive: bool = True,
) -> MultiRobotDisplayState:
```

```python
def update_multi_display(
    mds: MultiRobotDisplayState,
    desired_states: List[TrajectoryPoint],
    vehicle_states: List[VehicleState],
    pause: float = 0.001,
    interactive: bool = True,
) -> None:
```

```python
def save_multi_display(mds: MultiRobotDisplayState, filepath: str) -> None:
def close_multi_display(mds: MultiRobotDisplayState) -> None:
```

```python
def animate_multi_display(
    config: MultiRobotSimConfig,
    paths: List[Path],
    trajectories: List[Trajectory],
    history: List[List[Tuple[TrajectoryPoint, VehicleState]]],
    filepath: str = "sim_multi_animation.gif",
    fps: int = 20,
    step: int = 1,
) -> None:
```

`history[i]` is the ordered list of `(desired, vehicle)` pairs recorded for robot `i` during the simulation.

---

## Running the Tests

The test suite uses [pytest](https://docs.pytest.org/):

```bash
pytest tests/ -v
```

All tests are in `tests/test_sim.py` and cover obstacles, dynamics, trajectory, controller, the planner (integration), and multi-robot types and visualizer functions.

