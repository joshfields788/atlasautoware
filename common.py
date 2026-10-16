"""Shared constants, env construction, preprocessing and the episode loop.

Targets F1TENTH gym v1.0 (Gymnasium API): gym.make("f1tenth_gym:f1tenth-v0").
Action per step is np.array([[steering_angle, speed]]) (shape: num_agents x 2).
"""
import numpy as np
import gymnasium as gym

# --- LiDAR / vehicle constants (gym defaults) ---
FOV = 4.7                # rad (~270 deg)
NUM_BEAMS = 1080
STEER_MAX = 0.4189       # rad, gym's s_max
SPEED_MAX = 6.0          # m/s cap we command (gym allows 20, but keep it sane)

# --- Policy input ---
LIDAR_CLIP = 10.0        # m, ranges beyond this are clipped
POOL = 10                # 1080 beams -> 108 features via min-pooling
N_FEATURES = NUM_BEAMS // POOL

TRAIN_MAPS = ["Spielberg", "Austin", "Monza", "Catalunya", "Budapest"]
TEST_MAPS = ["Silverstone", "Oschersleben", "Zandvoort"]  # never trained on


def make_env(map_name, reset_type="rl_random_static", render_mode=None):
    config = {
        "map": map_name,
        "num_agents": 1,
        "observation_config": {
            "type": "features",
            "features": ["scan", "linear_vel_x", "collision", "lap_time", "lap_count"],
        },
        "reset_config": {"type": reset_type},
    }
    return gym.make("f1tenth_gym:f1tenth-v0", config=config, render_mode=render_mode)


def unpack(obs):
    o = obs["agent_0"]
    return (np.asarray(o["scan"], dtype=np.float32), float(o["linear_vel_x"]),
            bool(o["collision"]), float(o["lap_count"]), float(o["lap_time"]))


def preprocess(scan):
    """(1080,) raw ranges -> (108,) in [0,1]. Min-pooling keeps the nearest
    obstacle in each sector, which is the safe thing to preserve."""
    s = np.clip(scan, 0.0, LIDAR_CLIP)
    return (s.reshape(N_FEATURES, POOL).min(axis=1) / LIDAR_CLIP).astype(np.float32)


def policy_input(scan, speed):
    return np.append(preprocess(scan), speed / SPEED_MAX).astype(np.float32)


def run_episode(env, policy, max_time=120.0, seed=None, noise=0.0, rng=None, record=False):
    """Run one episode. If noise>0, random steering bursts are injected into the
    *executed* action while the *recorded label* stays the clean policy output
    (DART-style): this teaches a cloned policy how to recover from mistakes."""
    rng = rng or np.random.default_rng()
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    X, Y = [], []
    burst_left, burst_val = 0, 0.0
    collided, term, lap_time, laps = False, False, 0.0, 0.0
    while True:
        scan, v, collided, laps, lap_time = unpack(obs)
        steer, speed = policy.act(scan, v)
        if record:
            X.append(policy_input(scan, v))
            Y.append([steer / STEER_MAX, speed / SPEED_MAX])
        exec_steer = steer
        if noise > 0:
            if burst_left == 0 and rng.random() < 0.01:
                burst_left, burst_val = int(rng.integers(10, 30)), rng.normal(0, noise)
            if burst_left > 0:
                exec_steer = float(np.clip(steer + burst_val, -STEER_MAX, STEER_MAX))
                burst_left -= 1
        obs, _, term, trunc, _ = env.step(np.array([[exec_steer, speed]]))
        if term or trunc or env.unwrapped.current_time > max_time:
            break
    scan, v, collided, laps, lap_time = unpack(obs)
    out = {"collided": collided, "finished": bool(term and not collided),
           "time": float(env.unwrapped.current_time), "laps": laps}
    if record:
        out["X"], out["Y"] = np.array(X, np.float32), np.array(Y, np.float32)
    return out
