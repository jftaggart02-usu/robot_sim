"""
robot_sim — modular mobile-robot simulation framework.

Packages are split by concern following data-oriented design:
  types       — shared dataclasses (pure data, no logic)
  obstacles   — polygon obstacle helpers
  planners    — planning implementations (RRT*)
  trajectory  — time-indexed trajectory generation
  controllers — control implementations (PID)
  dynamics    — unicycle-model dynamics
  visualizer  — matplotlib real-time display
"""
