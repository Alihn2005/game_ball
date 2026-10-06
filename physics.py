# physics.py
#
# Pure simulation - no pygame, no OpenCV, no drawing. Everything the
# viewer sees (trails, glow, sparks, squash) is derived from this
# module's state and the events it emits, in effects.py.
#
# What is simulated:
#   - balls falling under gravity, with slightly different sizes
#     (bigger = heavier),
#   - bouncing off detected stickers, which are ROTATED rectangles:
#     the bounce uses the real surface normal (flat edge or corner)
#     of the tilted sticker, with restitution (how bouncy) and a
#     little friction (so balls roll along slopes instead of
#     skating),
#   - balls bouncing off each other,
#   - the screen's side walls, and falling out of the bottom,
#   - "stuck" balls (parked in a pocket between stickers): they stay
#     for a while, then pop out of existence, freeing their slot.
#
# Performance notes (this runs 144 times a second):
#   - obstacle dicts are converted ONCE per detection update into
#     lightweight _Solid records with cos/sin already computed,
#   - each ball first does a cheap circle-vs-circle reject against
#     each solid's bounding circle before any real collision math,
#   - time is split into equal slices no longer than
#     PHYSICS_MAX_STEP_SECONDS (and short enough that a fast ball
#     can't skip over a sticker), so motion stays smooth and
#     proportional to real elapsed time.

import math
import random

from geometry import (
    extract_obb,
    obb_to_segments,
    build_boundary_segments,
    swept_ball_vs_segment,
    reflect,
    Segment,
)

from config import (
    BALL_RADIUS,
    BALL_RADIUS_VARIATION,
    BALL_GRAVITY,
    BALL_RESTITUTION,
    BALL_REST_SPEED,
    BALL_FRICTION,
    BALL_MAX_SPEED,
    BALL_BALL_COLLISION_ENABLED,
    BALL_BALL_RESTITUTION,
    PHYSICS_MAX_STEP_SECONDS,
    MAX_SIMULTANEOUS_BALLS,
    BALL_SPAWN_RANDOM_X,
    BALL_SPAWN_MARGIN,
    BALL_SPAWN_VELOCITY_JITTER,
    BALL_STUCK_DISTANCE,
    BALL_STUCK_TIMEOUT_SECONDS,
    BALL_MAX_LIFETIME_SECONDS,
    BALL_HUE_MIN_DEGREES,
    BALL_HUE_MAX_DEGREES,
    IMPACT_EFFECT_MIN_SPEED,
    LAUNCHER_ENABLED,
    LAUNCHER_SWEEP_ANGLE_DEGREES,
    LAUNCHER_SWEEP_SPEED,
    LAUNCHER_BARREL_LENGTH,
    LAUNCHER_BARREL_WIDTH,
    LAUNCH_BALL_SPEED,
)



_ROLLING_DRAG = 0.2
_DYING_SECONDS = 0.35
_MAX_EVENTS_PER_FRAME = 64


class Ball:
    def __init__(
        self,
        x,
        y,
        radius=16,
        velocity_x=0.0,
        velocity_y=0.0,
        gravity=900.0,
        hue=200.0,
    ):
        self.x = float(x)
        self.y = float(y)

        self.radius = float(radius)
        self.mass = self.radius * self.radius

        self.velocity_x = float(velocity_x)
        self.velocity_y = float(velocity_y)

        self.gravity = float(gravity)

        self.alive = True
        self.dying = False
        self.fade = 1.0

        self.age = 0.0

        # Cosmetic state owned by effects.py
        self.hue = float(hue)
        self.trail = []
        self.trail_timer = 0.0
        self.squash = 0.0
        self.squash_nx = 0.0
        self.squash_ny = -1.0

        # Stuck detection
        self._anchor_x = self.x
        self._anchor_y = self.y
        self._stuck_time = 0.0

    @property
    def speed(self):
        return math.hypot(self.velocity_x, self.velocity_y)


class PhysicsWorld:
    def __init__(self, width, height):
        self.width = float(width)
        self.height = float(height)

        self.balls = []
        self.events = []

        self._obstacle_segments = []
        self._solid_source = None

        # Launcher state
        self.launcher_time = 0.0
        self.launcher_angle = 0.0

        # Build closed outer boundary segments around the screen
        self._boundary_segments = build_boundary_segments(self.width, self.height)

        self._random = random.Random()
        self._hue_cursor = self._random.uniform(
            BALL_HUE_MIN_DEGREES,
            BALL_HUE_MAX_DEGREES,
        )

    def get_nozzle_transform(self):
        """
        Returns (nozzle_x, nozzle_y, dir_x, dir_y, angle_rad) for launcher.
        Pivot is at top center: (self.width / 2.0, 0.0).
        """
        angle = self.launcher_angle
        pivot_x = self.width / 2.0
        pivot_y = 0.0
        dir_x = math.sin(angle)
        dir_y = math.cos(angle)
        nozzle_x = pivot_x + LAUNCHER_BARREL_LENGTH * dir_x
        nozzle_y = pivot_y + LAUNCHER_BARREL_LENGTH * dir_y
        return nozzle_x, nozzle_y, dir_x, dir_y, angle

    def get_launcher_segments(self):
        """
        Constructs the 4 physical collision segments for the rotating launcher tube.
        """
        if not LAUNCHER_ENABLED:
            return []
        cx = self.width / 2.0
        cy = 0.0
        angle = self.launcher_angle
        sin_a = math.sin(angle)
        cos_a = math.cos(angle)
        center_offset_y = (LAUNCHER_BARREL_LENGTH - 20.0) / 2.0
        total_length = LAUNCHER_BARREL_LENGTH + 20.0
        center_x = cx + center_offset_y * sin_a
        center_y = cy + center_offset_y * cos_a
        angle_deg = math.degrees(angle)
        return obb_to_segments(center_x, center_y, total_length, LAUNCHER_BARREL_WIDTH, angle_deg)

    # ========================================================
    # Balls
    # ========================================================

    def add_ball(self, ball):
        if ball is not None:
            self.balls.append(ball)

    def live_ball_count(self):
        return sum(
            1
            for ball in self.balls
            if ball.alive and not ball.dying
        )

    def _next_hue(self):
        span = BALL_HUE_MAX_DEGREES - BALL_HUE_MIN_DEGREES
        self._hue_cursor = (
            BALL_HUE_MIN_DEGREES
            + (
                self._hue_cursor
                - BALL_HUE_MIN_DEGREES
                + span * 0.618034
            )
            % span
        )
        return self._hue_cursor

    def spawn_ball(self):
        """
        Spawns a new ball directly from the launcher nozzle.
        Returns the ball, or None if blocked or too many balls already.
        """
        if self.live_ball_count() >= MAX_SIMULTANEOUS_BALLS:
            return None

        rng = self._random

        radius = BALL_RADIUS * (
            1.0
            + rng.uniform(
                -BALL_RADIUS_VARIATION,
                BALL_RADIUS_VARIATION,
            )
        )

        if LAUNCHER_ENABLED:
            nx, ny, dx, dy, angle = self.get_nozzle_transform()
            # Spawn ball slightly outside nozzle tip so it doesn't collide with nozzle
            spawn_offset = radius + 4.0
            spawn_x = nx + dx * spawn_offset
            spawn_y = ny + dy * spawn_offset

            vx = dx * LAUNCH_BALL_SPEED
            vy = dy * LAUNCH_BALL_SPEED

            ball = Ball(
                x=spawn_x,
                y=spawn_y,
                radius=radius,
                velocity_x=vx,
                velocity_y=vy,
                gravity=BALL_GRAVITY,
                hue=self._next_hue(),
            )

            self.balls.append(ball)

            if len(self.events) < _MAX_EVENTS_PER_FRAME:
                self.events.append(
                    (
                        "launch",
                        nx,
                        ny,
                        dx,
                        dy,
                        LAUNCH_BALL_SPEED,
                        ball,
                    )
                )

            return ball

        for _ in range(8):
            if BALL_SPAWN_RANDOM_X:
                x = rng.uniform(
                    BALL_SPAWN_MARGIN,
                    self.width - BALL_SPAWN_MARGIN,
                )
            else:
                x = self.width / 2.0

            y = radius * 2.0

            blocked = False

            for other in self.balls:
                if not other.alive or other.dying:
                    continue

                min_distance = (radius + other.radius) * 1.05

                if math.hypot(x - other.x, y - other.y) < min_distance:
                    blocked = True
                    break

            if not blocked:
                ball = Ball(
                    x=x,
                    y=y,
                    radius=radius,
                    velocity_x=rng.uniform(
                        -BALL_SPAWN_VELOCITY_JITTER,
                        BALL_SPAWN_VELOCITY_JITTER,
                    ),
                    velocity_y=50.0,
                    gravity=BALL_GRAVITY,
                    hue=self._next_hue(),
                )

                self.balls.append(ball)
                return ball

        return None

    # ========================================================
    # Obstacles
    # ========================================================

    def set_obstacles(self, obstacles):
        if obstacles is self._solid_source:
            return

        self._solid_source = obstacles
        obstacle_segments = []

        for obstacle in obstacles or ():
            obb = extract_obb(obstacle)
            if obb is None:
                continue

            cx, cy, long_side, short_side, angle = obb
            segs = obb_to_segments(cx, cy, long_side, short_side, angle)
            obstacle_segments.extend(segs)

        self._obstacle_segments = obstacle_segments

    # ========================================================
    # Update
    # ========================================================

    def update(self, delta_time, obstacles=None):
        self.events = []

        if delta_time <= 0:
            return

        delta_time = min(delta_time, 0.05)

        if LAUNCHER_ENABLED:
            self.launcher_time += delta_time
            max_rad = math.radians(LAUNCHER_SWEEP_ANGLE_DEGREES)
            self.launcher_angle = max_rad * math.sin(
                2.0 * math.pi * LAUNCHER_SWEEP_SPEED * self.launcher_time
            )

        self.set_obstacles(obstacles)

        movers = [
            ball for ball in self.balls if ball.alive and not ball.dying
        ]

        if movers:
            fastest = max(ball.speed for ball in movers)
            smallest = min(ball.radius for ball in movers)

            steps = max(
                math.ceil(delta_time / PHYSICS_MAX_STEP_SECONDS),
                math.ceil(fastest * delta_time / (0.5 * smallest)),
                1,
            )
            steps = min(steps, 10)
            step = delta_time / steps

            for _ in range(steps):
                self._step(movers, step)

        self._housekeeping(delta_time)

    def _step(self, movers, h):
        all_segments = (
            self._obstacle_segments
            + self._boundary_segments
            + self.get_launcher_segments()
        )


        for ball in movers:
            ball.velocity_y += ball.gravity * h

            speed = ball.speed
            if speed > BALL_MAX_SPEED:
                scale = BALL_MAX_SPEED / speed
                ball.velocity_x *= scale
                ball.velocity_y *= scale

            # Continuous swept integration against segments
            dx = ball.velocity_x * h
            dy = ball.velocity_y * h
            ox, oy = ball.x, ball.y
            remaining = 1.0

            for _ in range(6):
                if remaining <= 1e-6:
                    break

                vx_step = dx * remaining
                vy_step = dy * remaining

                best = (False, 1.0, 0.0, 0.0, None)

                for seg in all_segments:
                    hit, t, nx, ny = swept_ball_vs_segment(
                        ox, oy, vx_step, vy_step, ball.radius, seg
                    )
                    if hit and t < best[1]:
                        best = (True, t, nx, ny, seg)

                if not best[0]:
                    ox += vx_step
                    oy += vy_step
                    remaining = 0.0
                    break

                _, t, nx, ny, _seg = best

                # Move to contact point slightly before contact
                t_contact = max(0.0, t - 1e-6)
                ox += vx_step * t_contact
                oy += vy_step * t_contact

                # Elastic reflection across surface normal
                rx, ry = reflect(ball.velocity_x, ball.velocity_y, nx, ny)
                ball.velocity_x = rx * BALL_RESTITUTION
                ball.velocity_y = ry * BALL_RESTITUTION

                impact = ball.speed

                if (
                    impact >= IMPACT_EFFECT_MIN_SPEED
                    and len(self.events) < _MAX_EVENTS_PER_FRAME
                ):
                    self.events.append(
                        (
                            "obstacle",
                            ox - nx * ball.radius,
                            oy - ny * ball.radius,
                            nx,
                            ny,
                            impact,
                            ball,
                        )
                    )

                remaining *= (1.0 - t)
                dx = ball.velocity_x * h
                dy = ball.velocity_y * h

            ball.x, ball.y = ox, oy
            self._keep_inside_walls(ball)

        if BALL_BALL_COLLISION_ENABLED and len(movers) > 1:
            self._collide_balls(movers)

    # ========================================================
    # Ball vs. ball
    # ========================================================

    def _collide_balls(self, movers):
        count = len(movers)

        for i in range(count - 1):
            first = movers[i]

            for j in range(i + 1, count):
                second = movers[j]

                reach = first.radius + second.radius
                dx = second.x - first.x
                dy = second.y - first.y

                if abs(dx) >= reach or abs(dy) >= reach:
                    continue

                distance_sq = dx * dx + dy * dy
                if distance_sq >= reach * reach:
                    continue

                distance = math.sqrt(distance_sq)
                if distance > 1e-6:
                    normal_x = dx / distance
                    normal_y = dy / distance
                else:
                    normal_x = 0.0
                    normal_y = 1.0

                total_mass = first.mass + second.mass

                penetration = reach - distance
                first_share = penetration * (second.mass / total_mass)
                second_share = penetration * (first.mass / total_mass)

                first.x -= normal_x * first_share
                first.y -= normal_y * first_share
                second.x += normal_x * second_share
                second.y += normal_y * second_share

                closing = (
                    (second.velocity_x - first.velocity_x) * normal_x
                    + (second.velocity_y - first.velocity_y) * normal_y
                )

                if closing >= 0:
                    continue

                impact = -closing
                bounce = (
                    BALL_BALL_RESTITUTION
                    if impact >= BALL_REST_SPEED
                    else 0.0
                )

                impulse = (
                    -(1.0 + bounce)
                    * closing
                    / (1.0 / first.mass + 1.0 / second.mass)
                )

                first.velocity_x -= impulse / first.mass * normal_x
                first.velocity_y -= impulse / first.mass * normal_y
                second.velocity_x += impulse / second.mass * normal_x
                second.velocity_y += impulse / second.mass * normal_y

                if (
                    impact >= IMPACT_EFFECT_MIN_SPEED
                    and len(self.events) < _MAX_EVENTS_PER_FRAME
                ):
                    self.events.append(
                        (
                            "ball",
                            first.x + normal_x * first.radius,
                            first.y + normal_y * first.radius,
                            normal_x,
                            normal_y,
                            impact,
                            first,
                        )
                    )

    # ========================================================
    # Closed Outer Screen Boundary (Top, Bottom, Left, Right)
    # ========================================================

    def _keep_inside_walls(self, ball):
        radius = ball.radius

        # Left wall
        if ball.x - radius < 0:
            ball.x = radius
            if ball.velocity_x < 0:
                ball.velocity_x = -ball.velocity_x * BALL_RESTITUTION

        # Right wall
        elif ball.x + radius > self.width:
            ball.x = self.width - radius
            if ball.velocity_x > 0:
                ball.velocity_x = -ball.velocity_x * BALL_RESTITUTION

        # Top wall
        if ball.y - radius < 0:
            ball.y = radius
            if ball.velocity_y < 0:
                ball.velocity_y = -ball.velocity_y * BALL_RESTITUTION

        # Bottom wall
        elif ball.y + radius > self.height:
            ball.y = self.height - radius
            if ball.velocity_y > 0:
                ball.velocity_y = -ball.velocity_y * BALL_RESTITUTION

    # ========================================================
    # Housekeeping & Lifetime Management
    # ========================================================

    def _housekeeping(self, delta_time):
        for ball in list(self.balls):
            if not ball.alive:
                continue

            ball.age += delta_time

            # Check stuck timeout
            if BALL_STUCK_TIMEOUT_SECONDS > 0:
                moved = math.hypot(
                    ball.x - ball._anchor_x,
                    ball.y - ball._anchor_y,
                )

                if moved > BALL_STUCK_DISTANCE:
                    ball._anchor_x = ball.x
                    ball._anchor_y = ball.y
                    ball._stuck_time = 0.0
                else:
                    ball._stuck_time += delta_time

                if ball._stuck_time >= BALL_STUCK_TIMEOUT_SECONDS:
                    self._destroy_ball(ball)
                    continue

            # Check configurable lifetime
            if (
                BALL_MAX_LIFETIME_SECONDS > 0
                and ball.age >= BALL_MAX_LIFETIME_SECONDS
            ):
                self._destroy_ball(ball)

        # Remove dead balls from active physics memory instantly
        self.balls = [ball for ball in self.balls if ball.alive]

    def _destroy_ball(self, ball):
        """
        Instantly remove physics object and emit pop effect event.
        The visual pop effect runs independently in effects.py.
        """
        ball.alive = False

        if len(self.events) < _MAX_EVENTS_PER_FRAME:
            self.events.append(
                (
                    "pop",
                    ball.x,
                    ball.y,
                    0.0,
                    -1.0,
                    0.0,
                    ball,
                )
            )

    # ========================================================
    # Reset
    # ========================================================

    def clear(self):
        self.balls = []
        self.events = []

    def reset(self):
        self.clear()

