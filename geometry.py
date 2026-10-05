# geometry.py
#
# Small, shared helpers for working with ROTATED rectangles (an
# "OBB" - oriented bounding box) in screen space: a rectangle
# stickers can be glued on crooked, so an obstacle's position is no
# longer just an axis-aligned (x, y, width, height) - it also has
# an angle. This module is the single place that knows how to:
#
#   - turn a raw cv2.minAreaRect() result into a STABLE, canonical
#     (center, long side, short side, angle) representation that
#     doesn't jitter or jump between frames (see normalize_angle
#     and detection.py for why that's needed),
#   - blend/interpolate that angle smoothly across frames
#     (blend_angle),
#   - convert between world space and the rectangle's own local,
#     axis-aligned frame (world_to_local / local_to_world) - this
#     is what lets both collision detection AND collision response
#     be written as plain axis-aligned math that "just works" for
#     any rotation, by doing the rotation once at the boundary
#     instead of re-deriving rotated formulas everywhere,
#   - compute the 4 corners of a rectangle for drawing
#     (obb_corners),
#   - pull all of the above out of a detected-obstacle dict in one
#     place, with one shared definition of what counts as a valid
#     obstacle (extract_obb), so detection/physics/rendering can
#     never disagree about it.
#
# Angle convention: an un-rotated rectangle (long side horizontal)
# has angle 0. Positive angles rotate the long side counter-
# clockwise in standard math coordinates - visually CLOCKWISE on
# screen, since screen y grows downward. Angles are always kept in
# the canonical range (-90, 90], because a rectangle looks
# identical after a 180-degree rotation (it has no "arrow" printed
# on it), so anything outside that range is a redundant duplicate
# of an angle already inside it. Collapsing to one canonical range
# is what keeps blend_angle's shortest-path interpolation well
# defined instead of it occasionally spinning the "wrong" way.

import math


# ============================================================
# Angle canonicalization
# ============================================================

def normalize_angle(angle_degrees, period=180.0):
    """
    Fold an orientation angle into the canonical (-period/2,
    period/2] range.

    A rectangle has 180-degree rotational symmetry, so any angle is
    equivalent to one inside a 180-degree-wide window (period=180).
    A (nearly) SQUARE sticker is even more symmetric - it looks the
    same after every 90 degrees - so for those callers pass
    period=90. Always folding into the SAME window is what makes
    two angles describing the same physical orientation compare and
    blend as equal instead of looking like they are far apart.
    """
    half = period / 2.0

    angle = angle_degrees % period

    if angle > half:
        angle -= period

    return angle


def blend_angle(old_angle, new_angle, alpha, period=180.0):
    """
    Move old_angle toward new_angle by fraction alpha along the
    SHORTEST path, for orientations that repeat every `period`
    degrees.

    A naive `old + alpha * (new - old)` breaks near the wrap-around
    point: 89 degrees and -89 degrees are really only 2 degrees
    apart for a rectangle, but the naive subtraction sees 178 and
    blends the wrong way (and for a near-square sticker, averaging
    across its 90-degree ambiguity would even produce a completely
    wrong in-between angle). Wrapping the raw difference into
    (-period/2, period/2] first avoids both.
    """
    half = period / 2.0

    diff = ((new_angle - old_angle + half) % period) - half

    return normalize_angle(old_angle + alpha * diff, period)


def paint_dimensions(rect_long, rect_short, inset_ratio):
    """
    Size of the white patch painted on the monitor for a detected
    sticker: the sticker trimmed by inset_ratio * short side on every
    edge. Shared by rendering (to draw it) and detection (to blank
    exactly that area out before looking for stickers) so the two can
    never disagree about what was painted.
    """
    inset = inset_ratio * rect_short

    return (
        max(1.0, rect_long - 2.0 * inset),
        max(1.0, rect_short - 2.0 * inset),
    )


# ============================================================
# World <-> local (rectangle-aligned) frame conversion
# ============================================================

def world_to_local(x, y, angle_degrees):
    """
    Rotate a world-space vector (x, y) by -angle_degrees, i.e. into
    the rectangle's own local frame, where its long side lies along
    the local x-axis and its short side along the local y-axis.

    This takes a VECTOR, not a point - to transform a point, first
    subtract the rectangle's center, call this, and the result is
    the point's position relative to the rectangle in its own
    frame. The same function (with no subtraction) also correctly
    rotates direction-only quantities like velocity, since rotation
    doesn't care about position.
    """
    theta = math.radians(angle_degrees)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    return (
        x * cos_t + y * sin_t,
        -x * sin_t + y * cos_t,
    )


def local_to_world(x, y, angle_degrees):
    """
    The exact inverse of world_to_local: rotate a local-frame
    vector back by +angle_degrees into world space.
    """
    theta = math.radians(angle_degrees)

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    return (
        x * cos_t - y * sin_t,
        x * sin_t + y * cos_t,
    )


# ============================================================
# Corners (for drawing / debugging)
# ============================================================

def obb_corners(center_x, center_y, rect_long, rect_short, angle_degrees):
    """
    Return the 4 corners of a rotated rectangle, in order around
    its perimeter, as world-space (x, y) tuples. Safe to hand
    straight to pygame.draw.polygon or cv2.polylines.
    """
    half_long = rect_long / 2.0
    half_short = rect_short / 2.0

    local_corners = (
        (half_long, half_short),
        (half_long, -half_short),
        (-half_long, -half_short),
        (-half_long, half_short),
    )

    corners = []

    for local_x, local_y in local_corners:
        world_dx, world_dy = local_to_world(
            local_x,
            local_y,
            angle_degrees,
        )

        corners.append(
            (
                center_x + world_dx,
                center_y + world_dy,
            )
        )

    return corners


# ============================================================
# Obstacle dict -> OBB extraction
# ============================================================

def extract_obb(obstacle):
    """
    Pull (center_x, center_y, rect_long, rect_short, angle) out of
    a detected-obstacle dict, or return None if the obstacle is
    missing/malformed/degenerate. This is the ONE place that
    decides what counts as a usable oriented rectangle, so physics,
    rendering, and validity-checking can never disagree with each
    other about it.

    Falls back to treating the obstacle as an axis-aligned
    (angle=0) rectangle built from x/y/width/height if it has no
    "center"/"rect_long"/"rect_short"/"angle" fields, so any code
    that still hands in a plain axis-aligned obstacle dict keeps
    working exactly as before.
    """
    if not isinstance(obstacle, dict):
        return None

    try:
        if "center" in obstacle:
            center_x, center_y = obstacle["center"]
        else:
            center_x = (
                obstacle["x"] + obstacle["width"] / 2.0
            )
            center_y = (
                obstacle["y"] + obstacle["height"] / 2.0
            )

        rect_long = obstacle.get(
            "rect_long",
            obstacle.get("width"),
        )
        rect_short = obstacle.get(
            "rect_short",
            obstacle.get("height"),
        )
        angle = obstacle.get("angle", 0.0)

        center_x = float(center_x)
        center_y = float(center_y)
        rect_long = float(rect_long)
        rect_short = float(rect_short)
        angle = float(angle)
    except (KeyError, TypeError, ValueError):
        return None

    if not all(
        math.isfinite(value)
        for value in (
            center_x,
            center_y,
            rect_long,
            rect_short,
            angle,
        )
    ):
        return None

    if rect_long <= 0 or rect_short <= 0:
        return None

    return center_x, center_y, rect_long, rect_short, angle


# ============================================================
# Continuous Collision Detection (CCD) helpers from version_1.py
# ============================================================

EPS = 1e-9


class Segment:
    """A directed 2D line segment between (x1, y1) and (x2, y2)."""

    __slots__ = ("x1", "y1", "x2", "y2")

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)

    @property
    def dx(self) -> float:
        return self.x2 - self.x1

    @property
    def dy(self) -> float:
        return self.y2 - self.y1


def dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


def unit(vx: float, vy: float) -> tuple[float, float]:
    length = math.hypot(vx, vy)
    return (vx / length, vy / length) if length > 0 else (0.0, 0.0)


def ray_hit_circle(
    ox: float, oy: float,
    vx: float, vy: float,
    cx: float, cy: float,
    r: float
) -> tuple[bool, float, float, float]:
    """
    Ray from O along displacement V intersects circle at C with radius r.
    Returns (hit, t, nx, ny) with t in [0, 1].
    Normal is outward from circle center at hit point.
    """
    fx = ox - cx
    fy = oy - cy

    a = dot(vx, vy, vx, vy)
    b = 2.0 * dot(fx, fy, vx, vy)
    c = dot(fx, fy, fx, fy) - r * r

    if a < EPS:
        return (False, 0.0, 0.0, 0.0)

    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return (False, 0.0, 0.0, 0.0)

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    t = None
    if 0.0 <= t1 <= 1.0:
        t = t1
    elif 0.0 <= t2 <= 1.0:
        t = t2

    if t is None:
        return (False, 0.0, 0.0, 0.0)

    hx = ox + vx * t
    hy = oy + vy * t
    nx, ny = unit(hx - cx, hy - cy)
    return (True, t, nx, ny)


def swept_ball_vs_segment(
    ox: float, oy: float,
    vx: float, vy: float,
    r: float,
    seg: Segment
) -> tuple[bool, float, float, float]:
    """
    Swept collision of moving ball center (O to O+V) vs capsule around segment (radius r).
    Returns earliest hit (hit, t, nx, ny) where normal points AWAY from segment surface.
    """
    ax, ay = seg.x1, seg.y1
    bx, by = seg.x2, seg.y2
    sx, sy = bx - ax, by - ay
    L = math.hypot(sx, sy)
    if L < EPS:
        return ray_hit_circle(ox, oy, vx, vy, ax, ay, r)

    tx, ty = sx / L, sy / L
    nx0, ny0 = -ty, tx

    relx, rely = ox - ax, oy - ay
    u0 = dot(relx, rely, tx, ty)
    d0 = dot(relx, rely, nx0, ny0)

    vu = dot(vx, vy, tx, ty)
    vd = dot(vx, vy, nx0, ny0)

    best_hit = (False, 1.0, 0.0, 0.0)

    # 1) Infinite strip lines
    if abs(vd) > EPS:
        for sign in (+1.0, -1.0):
            t = (sign * r - d0) / vd
            if 0.0 <= t <= 1.0:
                u = u0 + vu * t
                if 0.0 <= u <= L:
                    nx = sign * nx0
                    ny = sign * ny0
                    if t < best_hit[1]:
                        best_hit = (True, t, nx, ny)

    # 2) End caps
    h1 = ray_hit_circle(ox, oy, vx, vy, ax, ay, r)
    if h1[0] and h1[1] < best_hit[1]:
        best_hit = h1

    h2 = ray_hit_circle(ox, oy, vx, vy, bx, by, r)
    if h2[0] and h2[1] < best_hit[1]:
        best_hit = h2

    return best_hit


def reflect(vx: float, vy: float, nx: float, ny: float) -> tuple[float, float]:
    """Perfect elastic reflection of velocity vector across surface normal (nx, ny)."""
    vn = dot(vx, vy, nx, ny)
    return (vx - 2.0 * vn * nx, vy - 2.0 * vn * ny)


def obb_to_segments(center_x: float, center_y: float, rect_long: float, rect_short: float, angle_degrees: float) -> list[Segment]:
    """Convert an oriented rectangle obstacle into 4 directed boundary Segments."""
    corners = obb_corners(center_x, center_y, rect_long, rect_short, angle_degrees)
    return [
        Segment(corners[0][0], corners[0][1], corners[1][0], corners[1][1]),
        Segment(corners[1][0], corners[1][1], corners[2][0], corners[2][1]),
        Segment(corners[2][0], corners[2][1], corners[3][0], corners[3][1]),
        Segment(corners[3][0], corners[3][1], corners[0][0], corners[0][1]),
    ]


def build_boundary_segments(width: float, height: float, margin: float = 0.0) -> list[Segment]:
    """
    Construct 4 closed outer boundary segments around the screen.
    This creates an inescapable box for balls.
    """
    min_x = margin
    min_y = margin
    max_x = width - margin
    max_y = height - margin

    return [
        Segment(min_x, min_y, max_x, min_y),  # Top wall
        Segment(max_x, min_y, max_x, max_y),  # Right wall
        Segment(max_x, max_y, min_x, max_y),  # Bottom wall
        Segment(min_x, max_y, min_x, min_y),  # Left wall
    ]

