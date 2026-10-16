"""Record (LiDAR, speed) -> (steer, speed) pairs from the Follow-the-Gap expert."""
import argparse, os
import numpy as np
from common import make_env, run_episode, TRAIN_MAPS
from follow_the_gap import FollowTheGap

p = argparse.ArgumentParser()
p.add_argument("--maps", nargs="+", default=TRAIN_MAPS)
p.add_argument("--episodes", type=int, default=20, help="episodes per map")
p.add_argument("--noise", type=float, default=0.15, help="steering-burst std (rad); 0 = pure expert")
p.add_argument("--max-time", type=float, default=60.0)
p.add_argument("--out", default="data/ftg.npz")
a = p.parse_args()

rng, expert = np.random.default_rng(0), FollowTheGap()
Xs, Ys = [], []
for m in a.maps:
    env = make_env(m)
    ok = 0
    for ep in range(a.episodes):
        r = run_episode(env, expert, a.max_time, seed=ep, noise=a.noise, rng=rng, record=True)
        Xs.append(r["X"]); Ys.append(r["Y"]); ok += not r["collided"]
    print(f"{m}: {ok}/{a.episodes} episodes without collision, {sum(len(x) for x in Xs)} samples total")
    env.close()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
np.savez_compressed(a.out, X=np.concatenate(Xs), Y=np.concatenate(Ys))
print("saved", a.out)
