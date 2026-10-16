"""Compare policies on held-out maps: success rate (no collision, 2 laps) and time."""
import argparse
import numpy as np
from common import make_env, run_episode, TEST_MAPS
from follow_the_gap import FollowTheGap

p = argparse.ArgumentParser()
p.add_argument("--policy", default="ftg", help="'ftg' or path to a .pt file")
p.add_argument("--maps", nargs="+", default=TEST_MAPS)
p.add_argument("--episodes", type=int, default=5)
p.add_argument("--max-time", type=float, default=120.0)
p.add_argument("--render", action="store_true")
a = p.parse_args()

if a.policy == "ftg":
    policy = FollowTheGap()
else:
    from model import NeuralPolicy
    policy = NeuralPolicy(a.policy)

print(f"{'map':<14}{'success':>9}{'avg time (s)':>14}")
for m in a.maps:
    env = make_env(m, render_mode="human" if a.render else None)
    res = [run_episode(env, policy, a.max_time, seed=100 + i) for i in range(a.episodes)]
    done = [r["time"] for r in res if r["finished"]]
    print(f"{m:<14}{len(done):>4}/{a.episodes:<4}{np.mean(done) if done else float('nan'):>14.1f}")
    env.close()
