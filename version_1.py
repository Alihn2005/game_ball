from __future__ import annotations

from dataclasses import dataclass
import math
import pygame


EPS = 1e-9


def dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def unit(vx: float, vy: float) -> tuple[float, float]:
    l = math.hypot(vx, vy)
    return (vx / l, vy / l) if l > 0 else (0.0, 0.0)


def perp(vx: float, vy: float) -> tuple[float, float]:
    return -vy, vx


@dataclass
class Segment:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def dx(self) -> float:
        return self.x2 - self.x1

    @property
    def dy(self) -> float:
        return self.y2 - self.y1

    @property
    def normal_cw_unit(self) -> tuple[float, float]:
        # CW normal of segment direction
        return unit(-self.dy, self.dx)


@dataclass
class Rectangle:
    s1: Segment
    s2: Segment
    s3: Segment
    s4: Segment
    pts: list[tuple[float, float]]
    fill_color: tuple[int, int, int]
    border_color: tuple[int, int, int]

    @staticmethod
    def make(
        cx: float,
        cy: float,
        w: float,
        h: float,
        angle: float,
        fill_color: tuple[int, int, int],
        border_color: tuple[int, int, int],
        normals: str = "inward",  # "inward" for outer, "outward" for obstacles
    ) -> "Rectangle":
        hw = w / 2.0
        hh = h / 2.0
        c = math.cos(angle)
        s = math.sin(angle)

        local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        pts: list[tuple[float, float]] = []
        for lx, ly in local:
            x = cx + lx * c - ly * s
            y = cy + lx * s + ly * c
            pts.append((x, y))

        poly_pts = list(reversed(pts)) if normals == "outward" else list(pts)
        p0, p1, p2, p3 = poly_pts

        return Rectangle(
            s1=Segment(p0[0], p0[1], p1[0], p1[1]),
            s2=Segment(p1[0], p1[1], p2[0], p2[1]),
            s3=Segment(p2[0], p2[1], p3[0], p3[1]),
            s4=Segment(p3[0], p3[1], p0[0], p0[1]),
            pts=pts,
            fill_color=fill_color,
            border_color=border_color,
        )

    def segments(self) -> tuple[Segment, Segment, Segment, Segment]:
        return self.s1, self.s2, self.s3, self.s4

    def draw(self, surface: pygame.Surface, border_width: int = 4) -> None:
        draw_pts = [(round(x), round(y)) for x, y in self.pts]
        pygame.draw.polygon(surface, self.fill_color, draw_pts, 0)
        pygame.draw.polygon(surface, self.border_color, draw_pts, border_width)


@dataclass
class Ball:
    x: float
    y: float
    r: float
    color: tuple[int, int, int]
    vx: float
    vy: float
    g: float = 900.0  # px/s^2

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, self.color, (round(self.x), round(self.y)), int(self.r))


# ---------- Continuous collision (swept) helpers ----------

def ray_hit_circle(
    ox: float, oy: float,
    vx: float, vy: float,
    cx: float, cy: float,
    r: float
) -> tuple[bool, float, float, float]:
    """
    Ray from O along V intersects circle at C with radius r.
    Returns (hit, t, nx, ny) with t in [0,1] desired by caller.
    Normal is outward from circle at hit point.
    """
    # Solve |(O + tV) - C|^2 = r^2
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
    Swept collision of moving point (ball center) vs capsule around segment (radius r).
    Returns earliest hit (hit, t, nx, ny) where normal points AWAY from segment surface
    (i.e. direction to push the ball center).
    """
    ax, ay = seg.x1, seg.y1
    bx, by = seg.x2, seg.y2
    sx, sy = bx - ax, by - ay
    L = math.hypot(sx, sy)
    if L < EPS:
        return ray_hit_circle(ox, oy, vx, vy, ax, ay, r)

    tx, ty = sx / L, sy / L               # tangent unit
    nx0, ny0 = -ty, tx                    # one of the normals (perp of tangent)

    # Express motion in segment local coordinates:
    # coordinate along tangent (u), and along normal (d)
    # For a point P: u = dot(P-A, t), d = dot(P-A, n0)
    relx, rely = ox - ax, oy - ay
    u0 = dot(relx, rely, tx, ty)
    d0 = dot(relx, rely, nx0, ny0)

    vu = dot(vx, vy, tx, ty)
    vd = dot(vx, vy, nx0, ny0)

    best_hit = (False, 1.0, 0.0, 0.0)

    # --- 1) Hit the infinite strip lines d = +r or d = -r, then check u in [0,L]
    if abs(vd) > EPS:
        for sign in (+1.0, -1.0):
            t = (sign * r - d0) / vd
            if 0.0 <= t <= 1.0:
                u = u0 + vu * t
                if 0.0 <= u <= L:
                    # normal should push OUT of the segment surface:
                    # if we hit d = +r => normal is +n0 ; if d = -r => normal is -n0
                    nx = sign * nx0
                    ny = sign * ny0
                    if t < best_hit[1]:
                        best_hit = (True, t, nx, ny)

    # --- 2) Hit the end circles
    h1 = ray_hit_circle(ox, oy, vx, vy, ax, ay, r)
    if h1[0] and h1[1] < best_hit[1]:
        best_hit = h1

    h2 = ray_hit_circle(ox, oy, vx, vy, bx, by, r)
    if h2[0] and h2[1] < best_hit[1]:
        best_hit = h2

    return best_hit


def reflect(vx: float, vy: float, nx: float, ny: float) -> tuple[float, float]:
    # Perfectly elastic reflection
    vn = dot(vx, vy, nx, ny)
    return (vx - 2.0 * vn * nx, vy - 2.0 * vn * ny)


def integrate_with_ccd(ball: Ball, segments: list[Segment], dt: float, max_bounces: int = 6) -> None:
    """
    Move the ball under gravity with continuous collision detection.
    We do: apply acceleration to velocity (semi-implicit Euler), then sweep for collisions.
    """
    # update velocity from gravity
    ball.vy += ball.g * dt

    # desired displacement this frame
    dx = ball.vx * dt
    dy = ball.vy * dt

    ox, oy = ball.x, ball.y
    remaining = 1.0

    for _ in range(max_bounces):
        if remaining <= 1e-6:
            break

        vx_step = dx * remaining
        vy_step = dy * remaining

        best = (False, 1.0, 0.0, 0.0, None)  # hit,t,nx,ny,seg

        for seg in segments:
            hit, t, nx, ny = swept_ball_vs_segment(ox, oy, vx_step, vy_step, ball.r, seg)
            if hit and t < best[1]:
                best = (True, t, nx, ny, seg)

        if not best[0]:
            ox += vx_step
            oy += vy_step
            remaining = 0.0
            break

        _, t, nx, ny, _seg = best

        # move to contact point (slightly before to avoid re-hitting due to float errors)
        t_contact = max(0.0, t - 1e-6)
        ox += vx_step * t_contact
        oy += vy_step * t_contact

        # reflect velocity (elastic, no loss)
        ball.vx, ball.vy = reflect(ball.vx, ball.vy, nx, ny)

        # after reflection, recompute the full-frame displacement for the remaining time
        # remaining time fraction:
        remaining *= (1.0 - t)

        dx = ball.vx * dt
        dy = ball.vy * dt

    ball.x, ball.y = ox, oy


def collect_segments(rects: list[Rectangle]) -> list[Segment]:
    segs: list[Segment] = []
    for r in rects:
        segs.extend(list(r.segments()))
    return segs


# ---------- Demo main ----------

def main() -> None:
    pygame.init()
    W, H = 1200, 900
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    outer = Rectangle.make(
        cx=W / 2, cy=H / 2,
        w=W - 120, h=H - 120,
        angle=0.0,
        fill_color=(0, 0, 0),
        border_color=(255, 255, 0),
        normals="inward",
    )

    obstacle1 = Rectangle.make(
        cx=460, cy=420,
        w=280, h=80,
        angle=math.radians(18),
        fill_color=(255, 255, 255),
        border_color=(255, 255, 255),
        normals="outward",
    )

    obstacle2 = Rectangle.make(
        cx=780, cy=580,
        w=280, h=80,
        angle=math.radians(-16),
        fill_color=(255, 255, 255),
        border_color=(255, 255, 255),
        normals="outward",
    )

    rects = [outer, obstacle1, obstacle2]
    segments = collect_segments(rects)

    ball = Ball(
        x=450.0, y=120.0,
        r=16.0,
        color=(0, 190, 255),
        vx=60.0, vy=0.0,
        g=900.0,
    )

    running = True
    while running:
        dt = clock.tick(120) / 1000.0
        dt = min(dt, 0.033)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE:
                running = False

        integrate_with_ccd(ball, segments, dt, max_bounces=8)

        screen.fill((30, 30, 30))
        outer.draw(screen, border_width=6)
        obstacle1.draw(screen, border_width=4)
        obstacle2.draw(screen, border_width=4)
        ball.draw(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
