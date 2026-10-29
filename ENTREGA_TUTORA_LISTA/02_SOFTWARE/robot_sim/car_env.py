"""Entorno de auto 2D continuo para el agente de run_car.py.

Mapa 10x10, 3 obstaculos circulares (r=0.4) con layout reproducible por seed,
meta en (8.5, 8.5) con radio 0.8, inicio en (1.0, 1.0).

Estado continuo: (x, y, theta, v). Acciones discretas:
  0 = acelerar recto, 1 = avanzar en crucero,
  2/3 = giro suave izquierda/derecha,
  4/5 = giro cerrado izquierda/derecha,
  6/7 = frenar girando izquierda/derecha.

Sensores: 8 rayos (45 grados, rango 0-3), angulo y distancia a la meta, y
velocidad normalizada -> obs_dim = 11, codificada en 24 filas.

Reward: +10 por meta, hasta +1 total por episodio seguro, -10 por colision,
-5 por timeout y shaping por progreso a meta y seguridad.
"""
import numpy as np

MAP_SIZE = 10.0
V_MAX = 0.15
V_MIN = 0.03
OMEGA_MAX = 0.15
FRICTION = 0.05
V_ACCEL = 0.05
RAY_RANGE = 3.0
N_RAYS = 8
N_ACTIONS = 8
ACTION_NAMES = (
    "accelerate_straight",
    "coast_straight",
    "gentle_left",
    "gentle_right",
    "sharp_left",
    "sharp_right",
    "brake_left",
    "brake_right",
)
MAX_STEPS = 1000
GOAL_POS = (8.5, 8.5)
GOAL_R = 0.8           # curriculum: radio ampliado (era 0.5)
START_POS = (1.0, 1.0)
OBST_R = 0.4
N_OBST = 3             # curriculum: 3 obstaculos (era 6)
REWARD_STEP = -0.005       # coste temporal leve; no debe dominar el refuerzo
REWARD_GOAL = 10.0
REWARD_COLLISION = -10.0  # colision terminal con peso comparable a alcanzar meta
REWARD_TIMEOUT = -5.0     # no llegar sigue siendo peor que completar seguro
REWARD_SHAPING = 0.5      # curriculum: shaping x5 (era 0.1)
OBSTACLE_SAFE_DISTANCE = 1.0
REWARD_OBSTACLE_PROXIMITY = 0.60
FRONT_SAFE_DISTANCE = 1.5
REWARD_FRONT_PROXIMITY = 1.5
REWARD_CLEARANCE_PROGRESS = 1.0
SAFE_BONUS_PER_STEP = 0.02  # acumula +1 tras 50 pasos con margen seguro
SAFE_BONUS_CAP = 1.0
REWARD_ACCEL = 0.0        # DESACTIVADO: el bonus +0.02 sesgaba la politica
                          # hacia acelerar siempre (colisiones 260/300) y
                          # desacoplaba accion<->sensores. Se deja en 0.0.
# FIX 3 — shaping por apuntar: bonus/castigo angular inmediato.
REWARD_AIM_PROGRESS = 0.10
REWARD_STEERING_SMOOTHNESS = 0.02
TURN_PROGRESS_REQUIRED = 1.25
REWARD_UNPRODUCTIVE_TURN = 2.0
FULL_TURN_RADIANS = 2.0 * np.pi


class CarEnv:
    """Auto cinematico con sensores de rayo, reproducible por seed."""

    def __init__(self, seed=0):
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.obstacles = self._make_obstacles()
        self.obs_dim = 11
        self.n_actions = N_ACTIONS
        self.reset()

    def _make_obstacles(self):
        rng = np.random.default_rng(self.seed + 10_000)
        obstacles = []
        tries = 0
        while len(obstacles) < N_OBST and tries < 5000:
            tries += 1
            cx = float(rng.uniform(1.2, 8.8))
            cy = float(rng.uniform(1.2, 8.8))
            if np.hypot(cx - START_POS[0], cy - START_POS[1]) < 1.6:
                continue
            if np.hypot(cx - GOAL_POS[0], cy - GOAL_POS[1]) < 1.6:
                continue
            if all(np.hypot(cx - ox, cy - oy) > 2 * OBST_R + 0.6
                   for ox, oy in obstacles):
                obstacles.append((cx, cy))
        return obstacles

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
            self.obstacles = self._make_obstacles()
        self.x, self.y = START_POS
        # Curriculum: orientacion inicial apuntando a la meta ±30 grados,
        # en vez de uniforme. Reduce la fase de exploracion ciega.
        to_goal = np.arctan2(GOAL_POS[1] - self.y, GOAL_POS[0] - self.x)
        self.theta = float(to_goal + self.rng.uniform(-0.5, 0.5))
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi
        self.v = 0.0
        self.previous_steering = 0.0
        self.steering_reversals = 0
        self.steering_smoothness_penalty = 0.0
        self.turning_angle = 0.0
        self._turn_angle_since_progress = 0.0
        self._turn_progress_anchor_dist = self._dist_goal()
        self.unproductive_turns = 0
        self.unproductive_turn_penalty = 0.0
        self.path_length = 0.0
        self.steps = 0
        self._safe_bonus_total = 0.0
        self.trail = [(self.x, self.y)]
        self.prev_dist = self._dist_goal()
        sensors, self.ray_ends = self._sensors()
        self._last_sensors = sensors.copy()
        # Safety shaping uses the same ray readings available to the policy,
        # rather than hidden exact obstacle coordinates.
        self.prev_safety_potential = -self._sensor_hazard_from(sensors)
        self.prev_front_hazard = self._front_hazard_from(sensors)
        return sensors

    def _dist_goal(self):
        return float(np.hypot(GOAL_POS[0] - self.x, GOAL_POS[1] - self.y))

    def _goal_angle_error(self):
        aim = np.arctan2(GOAL_POS[1] - self.y, GOAL_POS[0] - self.x)
        return abs((aim - self.theta + np.pi) % (2 * np.pi) - np.pi)

    def safety_clearance(self):
        """Signed clearance to the nearest wall or obstacle, in map units."""
        wall = min(self.x, self.y, MAP_SIZE - self.x, MAP_SIZE - self.y)
        obstacle = min(
            (np.hypot(self.x - cx, self.y - cy) - OBST_R
             for cx, cy in self.obstacles),
            default=float("inf"),
        )
        return float(min(wall, obstacle))

    def front_clearance(self):
        # Rays 0 and +/-45 degrees are the forward safety sector.
        return float(min(
            np.hypot(self.x - self.ray_ends[i][0],
                     self.y - self.ray_ends[i][1])
            for i in (0, 1, 7)
        ))

    @staticmethod
    def _front_clearance_from(sensors):
        """Nearest distance in the forward sensor sector, in map units."""
        return float(min(sensors[0], sensors[1], sensors[7]) * RAY_RANGE)

    @staticmethod
    def _front_hazard_from(sensors):
        clearance = CarEnv._front_clearance_from(sensors)
        return float(np.clip(
            (FRONT_SAFE_DISTANCE - clearance) / FRONT_SAFE_DISTANCE,
            0.0, 1.0,
        ))

    @staticmethod
    def _nearest_sensor_clearance_from(sensors):
        """Nearest obstacle/wall distance reported by any of the eight rays."""
        return float(np.min(np.asarray(sensors[:N_RAYS])) * RAY_RANGE)

    @staticmethod
    def _sensor_hazard_from(sensors):
        """Orientation-smoothed proximity risk from the full 360-degree scan."""
        clearances = np.asarray(sensors[:N_RAYS], dtype=float) * RAY_RANGE
        per_ray = np.clip(
            (OBSTACLE_SAFE_DISTANCE - clearances) / OBSTACLE_SAFE_DISTANCE,
            0.0, 1.0,
        )
        # Rescale the average to retain a useful signal when a hazard spans
        # only one or two sectors, without a sharp nearest-ray orientation jump.
        return float(np.clip(np.mean(per_ray) * 4.0, 0.0, 1.0))

    def _raycast(self, theta):
        """Distancia desde el auto hasta el primer hit (pared u obstaculo)."""
        dx, dy = np.cos(theta), np.sin(theta)
        t_best = RAY_RANGE + 1.0
        # paredes del cuadro [0, MAP_SIZE]^2
        for t in ((MAP_SIZE - self.x) / dx if dx > 1e-9 else -1.0,
                  (0.0 - self.x) / dx if dx < -1e-9 else -1.0,
                  (MAP_SIZE - self.y) / dy if dy > 1e-9 else -1.0,
                  (0.0 - self.y) / dy if dy < -1e-9 else -1.0):
            if t >= 0:
                t_best = min(t_best, t)
        # obstaculos (interseccion rayo-circulo)
        for (cx, cy) in self.obstacles:
            fx, fy = self.x - cx, self.y - cy
            b = fx * dx + fy * dy
            c = fx * fx + fy * fy - OBST_R * OBST_R
            disc = b * b - c
            if disc < 0:
                continue
            t = -b - np.sqrt(disc)
            if 0 <= t < t_best:
                t_best = t
        return float(min(t_best, RAY_RANGE))

    def _sensors(self):
        sensors = np.zeros(self.obs_dim)
        ends = []
        for i in range(N_RAYS):
            th = self.theta + i * (2 * np.pi / N_RAYS)
            d = self._raycast(th)
            sensors[i] = d / RAY_RANGE
            ends.append((self.x + d * np.cos(th), self.y + d * np.sin(th)))
        # sensor 8: angulo relativo a la meta [-1, 1]
        ang = np.arctan2(GOAL_POS[1] - self.y, GOAL_POS[0] - self.x) - self.theta
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        sensors[8] = ang / np.pi
        # sensor 9: distancia a la meta normalizada [0, 1]
        sensors[9] = self._dist_goal() / (MAP_SIZE * np.sqrt(2))
        # sensor 10: velocidad actual normalizada; permite anticipar frenado.
        sensors[10] = np.clip(self.v / V_MAX, 0.0, 1.0)
        return sensors, ends

    def step(self, action):
        if not isinstance(action, (int, np.integer)) or not 0 <= action < N_ACTIONS:
            raise ValueError(f"action must be an integer in [0, {N_ACTIONS - 1}]")
        previous_hazard = self._sensor_hazard_from(self._last_sensors)
        previous_front_hazard = self._front_hazard_from(self._last_sensors)
        angle_before = self._goal_angle_error()
        steering = {
            2: 1.0,
            3: -1.0,
            4: 2.0,
            5: -2.0,
            6: 1.0,
            7: -1.0,
        }.get(action, 0.0)
        steering_change = steering - self.previous_steering
        steering_smoothness_penalty = (
            REWARD_STEERING_SMOOTHNESS * steering_change ** 2
        )
        steering_reversal = (
            self.previous_steering * steering < 0.0
        )
        if steering_reversal:
            self.steering_reversals += 1
        self.steering_smoothness_penalty += steering_smoothness_penalty
        self.previous_steering = steering
        heading_delta = 0.0
        if action == 0:
            self.v = min(V_MAX, self.v + V_ACCEL)
        elif action == 1:
            self.v = max(V_MIN, self.v)
        elif action in (2, 3):
            self.v = min(V_MAX, max(V_MIN, self.v) + 0.5 * V_ACCEL)
            heading_delta = OMEGA_MAX * (1 if action == 2 else -1)
            self.theta += heading_delta
        elif action in (4, 5):
            self.v = min(V_MAX, max(V_MIN, self.v) + 0.5 * V_ACCEL)
            heading_delta = 2.0 * OMEGA_MAX * (1 if action == 4 else -1)
            self.theta += heading_delta
        else:
            self.v = max(V_MIN, self.v - 0.5 * V_ACCEL)
            heading_delta = OMEGA_MAX * (1 if action == 6 else -1)
            self.theta += heading_delta
        self.turning_angle += abs(heading_delta)
        self._turn_angle_since_progress += abs(heading_delta)
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        previous_position = (self.x, self.y)
        self.x += self.v * np.cos(self.theta)
        self.y += self.v * np.sin(self.theta)
        self.path_length += float(np.hypot(
            self.x - previous_position[0], self.y - previous_position[1]
        ))
        self.v *= (1.0 - FRICTION)
        self.v = max(V_MIN, self.v)
        self.steps += 1
        self.trail.append((self.x, self.y))

        reward = REWARD_STEP
        if action == 0:
            reward += REWARD_ACCEL   # curiosidad: premia probar acelerar
        dist = self._dist_goal()
        reward += REWARD_SHAPING * (self.prev_dist - dist)
        self.prev_dist = dist
        unproductive_turn_penalty = 0.0
        while self._turn_angle_since_progress >= FULL_TURN_RADIANS:
            progress = self._turn_progress_anchor_dist - dist
            if progress < TURN_PROGRESS_REQUIRED:
                unproductive_turn_penalty += REWARD_UNPRODUCTIVE_TURN * (
                    1.0 - np.clip(
                        max(progress, 0.0) / TURN_PROGRESS_REQUIRED,
                        0.0,
                        1.0,
                    )
                )
                self.unproductive_turns += 1
            self._turn_angle_since_progress -= FULL_TURN_RADIANS
            self._turn_progress_anchor_dist = dist
        self.unproductive_turn_penalty += unproductive_turn_penalty
        reward -= unproductive_turn_penalty
        # FIX 3 — bonus/castigo por apuntar: girar hacia la meta da reward
        # inmediata sin esperar al final del episodio.
        angle_after = self._goal_angle_error()
        reward += REWARD_AIM_PROGRESS * (angle_before - angle_after)
        reward -= steering_smoothness_penalty

        done = False
        reason = "timeout"
        collision_wall = not (0.0 <= self.x <= MAP_SIZE
                              and 0.0 <= self.y <= MAP_SIZE)
        collision_obstacle = any(
            np.hypot(self.x - cx, self.y - cy) <= OBST_R
            for cx, cy in self.obstacles
        )
        if dist <= GOAL_R:                       # meta
            reward += REWARD_GOAL
            done, reason = True, "goal"
        elif collision_wall or collision_obstacle:
            reward += REWARD_COLLISION
            done, reason = True, "collision"
        elif self.steps >= MAX_STEPS:            # timeout
            reward += REWARD_TIMEOUT
            done, reason = True, "timeout"

        sensors, self.ray_ends = self._sensors()
        front_clearance = self._front_clearance_from(sensors)
        sensor_clearance = self._nearest_sensor_clearance_from(sensors)
        sensor_hazard = self._sensor_hazard_from(sensors)
        front_hazard = self._front_hazard_from(sensors)
        safety_clearance = self.safety_clearance()
        # A terminal collision can place the car outside the map, where
        # outward-facing rays no longer intersect the wall. Do not let that
        # invalid post-collision reading create a positive safety reward.
        reward_hazard = (
            max(previous_hazard, sensor_hazard)
            if reason == "collision" else sensor_hazard
        )
        safety_potential = -reward_hazard
        clearance_progress = safety_potential - self.prev_safety_potential
        self.prev_safety_potential = safety_potential
        avoidance_penalty = REWARD_OBSTACLE_PROXIMITY * reward_hazard
        reward += REWARD_CLEARANCE_PROGRESS * clearance_progress
        reward -= avoidance_penalty
        reward_front_hazard = (
            max(previous_front_hazard, front_hazard)
            if reason == "collision" else front_hazard
        )
        front_proximity_penalty = (
            REWARD_FRONT_PROXIMITY * reward_front_hazard ** 2
        )
        reward -= front_proximity_penalty
        self.prev_front_hazard = front_hazard
        safe_bonus = 0.0
        if reason != "collision" and front_clearance >= OBSTACLE_SAFE_DISTANCE:
            safe_bonus = min(
                SAFE_BONUS_PER_STEP,
                SAFE_BONUS_CAP - self._safe_bonus_total,
            )
            self._safe_bonus_total += safe_bonus
            reward += safe_bonus
        self._last_sensors = sensors.copy()
        info = {"reason": reason, "dist": dist, "steps": self.steps,
                "front_clearance": float(front_clearance),
                "sensor_front_clearance": float(front_clearance),
                "nearest_sensor_clearance": float(sensor_clearance),
                "sensor_hazard": float(sensor_hazard),
                "safety_clearance": float(safety_clearance),
                "collision_type": (
                    "wall+obstacle" if collision_wall and collision_obstacle
                    else "wall" if collision_wall
                    else "obstacle" if collision_obstacle else None
                ),
                "avoidance_penalty": float(avoidance_penalty),
                "front_hazard": float(front_hazard),
                "front_proximity_penalty": float(front_proximity_penalty),
                "clearance_progress": float(clearance_progress),
                "safe_bonus": float(safe_bonus),
                "steering": float(steering),
                "steering_change": float(steering_change),
                "steering_reversal": bool(steering_reversal),
                "steering_smoothness_penalty": float(
                    steering_smoothness_penalty
                ),
                "steering_reversals": int(self.steering_reversals),
                "total_steering_smoothness_penalty": float(
                    self.steering_smoothness_penalty
                ),
                "path_length": float(self.path_length)}
        info.update({
            "turning_angle": float(self.turning_angle),
            "unproductive_turns": int(self.unproductive_turns),
            "unproductive_turn_penalty": float(
                self.unproductive_turn_penalty
            ),
            "step_unproductive_turn_penalty": float(
                unproductive_turn_penalty
            ),
        })
        return sensors, float(reward), done, info

    def draw(self, ax, title="", result=None):
        """Mapa 2D: obstaculos, meta, auto (triangulo), rayos y trayectoria."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        ax.clear()
        ax.add_patch(Rectangle((0, 0), MAP_SIZE, MAP_SIZE,
                               fill=False, ec="black", lw=1.5))
        for (cx, cy) in self.obstacles:
            ax.add_patch(Circle((cx, cy), OBST_R,
                                facecolor="0.35", edgecolor="black"))
        ax.add_patch(Circle(GOAL_POS, GOAL_R,
                            facecolor="tab:green", edgecolor="darkgreen",
                            alpha=0.9))
        # trayectoria del episodio actual
        if len(self.trail) > 1:
            ts = np.asarray(self.trail)
            ax.plot(ts[:, 0], ts[:, 1], color="0.7", lw=1.4, ls="--")
        # rayos de sensores
        for (ex, ey) in getattr(self, "ray_ends", []):
            ax.plot([self.x, ex], [self.y, ey],
                    color="tab:orange", lw=0.8, alpha=0.45)
        # auto: triangulo orientado en theta
        tri = np.array([[1.1, 0.0], [-0.6, 0.65], [-0.6, -0.65]]) * 0.45
        R = np.array([[np.cos(self.theta), -np.sin(self.theta)],
                      [np.sin(self.theta), np.cos(self.theta)]])
        pts = tri @ R.T + np.array([self.x, self.y])
        ax.fill(pts[:, 0], pts[:, 1], facecolor="tab:blue",
                edgecolor="navy", lw=1.2, zorder=5)
        ax.set_xlim(-0.4, MAP_SIZE + 0.4)
        ax.set_ylim(-0.4, MAP_SIZE + 0.4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        color = {"goal": "tab:green", "collision": "tab:red"}.get(
            result, "tab:blue")
        ax.set_title(title, color=color, fontweight="bold", fontsize=10)
        # plasticos locales para que la GUI pueda actualizar bordes/tintes
        ax._car_border = ax.add_patch(
            Rectangle((0, 0), MAP_SIZE, MAP_SIZE, fill=False,
                      lw=4, edgecolor=color))
