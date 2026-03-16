"""
obstacles.py — pure functions for polygon obstacle collision detection.

Uses Shapely for robust computational geometry.
"""

from __future__ import annotations

from typing import List, Tuple

from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as ShapelyPolygon

from robot_sim.types import PolygonObstacle


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def to_shapely(obstacle: PolygonObstacle) -> ShapelyPolygon:
    """Convert a :class:`PolygonObstacle` to a Shapely polygon."""
    return ShapelyPolygon(obstacle.vertices)


# ---------------------------------------------------------------------------
# Collision queries
# ---------------------------------------------------------------------------

def point_in_obstacle(x: float, y: float, obstacle: PolygonObstacle) -> bool:
    """Return True if the 2-D point (x, y) lies inside *obstacle*."""
    return to_shapely(obstacle).contains(Point(x, y))


def point_in_any_obstacle(
    x: float, y: float, obstacles: List[PolygonObstacle]
) -> bool:
    """Return True if (x, y) lies inside any obstacle in *obstacles*."""
    pt = Point(x, y)
    return any(to_shapely(obs).contains(pt) for obs in obstacles)


def segment_collides_with_obstacle(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    obstacle: PolygonObstacle,
) -> bool:
    """Return True if the line segment p1→p2 intersects *obstacle*."""
    poly = to_shapely(obstacle)
    seg = LineString([p1, p2])
    return poly.intersects(seg)


def segment_collides_with_any(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    obstacles: List[PolygonObstacle],
) -> bool:
    """Return True if the segment p1→p2 intersects any obstacle."""
    seg = LineString([p1, p2])
    return any(to_shapely(obs).intersects(seg) for obs in obstacles)
