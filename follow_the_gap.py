"""Classical LiDAR-only expert: Follow-the-Gap. Used as (1) a lap-time baseline,
(2) the teacher for imitation learning, (3) a fallback controller."""
import numpy as np
from common import FOV, NUM_BEAMS, STEER_MAX


class FollowTheGap:
    def __init__(self, bubble=0.35, clip=4.0, gap_thresh=1.2, view_deg=180,
                 steer_gain=0.9, speed_min=1.5, speed_max=5.0):
        self.bubble, self.clip, self.gap_thresh = bubble, clip, gap_thresh
        self.steer_gain, self.speed_min, self.speed_max = steer_gain, speed_min, speed_max
        self.angles = np.linspace(-FOV / 2, FOV / 2, NUM_BEAMS)  # +angle = left
        self.res = FOV / (NUM_BEAMS - 1)
        self.view = np.abs(self.angles) <= np.deg2rad(view_deg) / 2

    def act(self, scan, speed=None):
        r = np.clip(np.asarray(scan, dtype=float), 0.0, self.clip)
        r = np.convolve(r, np.ones(5) / 5, mode="same")[self.view]
        ang = self.angles[self.view]
        raw = r.copy()

        # 1) zero a safety bubble around the closest point
        c = int(np.argmin(r))
        half = int(np.arctan2(self.bubble, max(r[c], 0.1)) / self.res)
        r[max(0, c - half): c + half + 1] = 0.0

        # 2) longest run of "free" beams
        free = (r > self.gap_thresh).astype(np.int8)
        d = np.diff(np.concatenate(([0], free, [0])))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        if len(starts) == 0:  # boxed in: crawl toward the most open direction
            t = int(np.argmax(raw))
            return float(np.clip(self.steer_gain * ang[t], -STEER_MAX, STEER_MAX)), self.speed_min * 0.5
        i = int(np.argmax(ends - starts))
        seg = r[starts[i]:ends[i]]
        top = np.where(seg >= 0.9 * seg.max())[0]       # deepest part of the gap
        target = starts[i] + int(top.mean())

        # 3) steer toward it; slow down for sharp turns and close obstacles
        steer = float(np.clip(self.steer_gain * ang[target], -STEER_MAX, STEER_MAX))
        base = self.speed_max - (self.speed_max - self.speed_min) * abs(steer) / STEER_MAX
        front = raw[np.abs(ang) < 0.1].min()
        return steer, float(min(base, max(self.speed_min, 1.5 * front)))
