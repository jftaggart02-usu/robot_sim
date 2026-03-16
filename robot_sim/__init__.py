"""
robot_sim — modular mobile-robot simulation framework.

Packages are split by concern following data-oriented design:
  types       — shared dataclasses (pure data, no logic)
  obstacles   — polygon obstacle helpers
  planner     — RRT* motion planner
  trajectory  — time-indexed trajectory generation
  controller  — PID controller
  dynamics    — unicycle-model dynamics
  visualizer  — matplotlib real-time display
"""
