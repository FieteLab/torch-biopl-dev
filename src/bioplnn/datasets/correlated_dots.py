import torch
from torch.utils.data import Dataset
import math


class CorrelatedDots(Dataset):
    def __init__(
        self,
        resolution=(128, 128),
        n_frames=10,
        n_dots=100,
        correlation=1.0,           # can be float or (low, high)
        max_speed=5,
        samples_per_epoch=10000,
        directions=None,           # list of direction indices to include (0-7)
        num_bars=0,                # number of vertical black bars to overlay
        bar_width=4,               # width in pixels of each vertical bar
        dot_shape='point',         # 'point', 'plus', or 'circle'
        dot_radius=3,              # arm length for 'plus'; ring radius for 'circle'
        jitter_speed=0.0,          # oscillatory jitter amplitude (float or (low, high))
        evidence_pulses=None,      # list of (start_t, duration, direction) tuples
    ):
        """
        Args:
            n_frames (int): Number of frames to generate.
            n_dots (int): Number of dots in each frame.
            resolution (tuple): Size of each frame (height, width).
            correlation (float or tuple): If float, fixed correlation for all samples.
                If tuple (low, high), each sample's correlation is drawn uniformly
                from [low, high].
            max_speed (int): Max pixels per frame.
            directions (list): Direction indices to include. None = all 8.
            num_bars (int): Number of vertical black bars evenly spaced across the frame.
            bar_width (int): Width in pixels of each vertical bar.
            dot_shape (str): Shape to render at each dot position.
                'point' - single pixel (default).
                'plus'  - cross shape with arm length dot_radius.
                'circle'- hollow circle ring of radius dot_radius.
            dot_radius (int): Controls size of 'plus' and 'circle' shapes.
            jitter_speed (float or tuple): Per-sample jitter amplitude. A random
                direction is chosen once per sample; jitter is added with alternating
                sign each frame so total displacement is zero. If tuple (low, high),
                amplitude is drawn uniformly per sample.
            evidence_pulses (list): List of (start_t, duration, direction) tuples.
                direction > 0: effective coherence is doubled (capped at 1.0).
                direction < 0: coherent motion direction is reversed.
                Multiple pulses may overlap.
        """
        self.n_frames = n_frames
        self.n_dots = n_dots
        self.resolution = resolution
        self.correlation = correlation
        self.max_speed = max_speed
        self.samples_per_epoch = samples_per_epoch
        self.num_bars = num_bars
        self.bar_width = bar_width
        self.dot_shape = dot_shape
        self.dot_radius = dot_radius
        self.jitter_speed = jitter_speed
        self.evidence_pulses = evidence_pulses or []

        # 24 directions in image coords where +x=right, +y=down.
        # Keys 0-7: cardinal and intercardinal (45° steps).
        # Keys 8-15: half-wind (22.5° offsets).
        # Keys 16-23: by-wind (15° offsets from cardinals).
        r2 = math.sqrt(2.0)
        a = math.cos(math.radians(22.5))   # ≈ 0.9239
        b = math.sin(math.radians(22.5))   # ≈ 0.3827
        c = math.cos(math.radians(15.0))   # ≈ 0.9659
        d = math.sin(math.radians(15.0))   # ≈ 0.2588
        self.direction_vectors = {
            0:  torch.tensor(( 1.0,  0.0)),          # E      (90°)
            1:  torch.tensor((-1.0,  0.0)),          # W      (270°)
            2:  torch.tensor(( 0.0, -1.0)),          # N      (0°)
            3:  torch.tensor(( 0.0,  1.0)),          # S      (180°)
            4:  torch.tensor(( 1.0, -1.0)) / r2,     # NE     (45°)
            5:  torch.tensor((-1.0, -1.0)) / r2,     # NW     (315°)
            6:  torch.tensor(( 1.0,  1.0)) / r2,     # SE     (135°)
            7:  torch.tensor((-1.0,  1.0)) / r2,     # SW     (225°)
            8:  torch.tensor(( a,   -b  )),           # ENE    (67.5°)
            9:  torch.tensor(( b,   -a  )),           # NNE    (22.5°)
            10: torch.tensor((-b,   -a  )),           # NNW    (337.5°)
            11: torch.tensor((-a,   -b  )),           # WNW    (292.5°)
            12: torch.tensor((-a,    b  )),           # WSW    (247.5°)
            13: torch.tensor((-b,    a  )),           # SSW    (202.5°)
            14: torch.tensor(( b,    a  )),           # SSE    (157.5°)
            15: torch.tensor(( a,    b  )),           # ESE    (112.5°)
            16: torch.tensor(( d,   -c  )),           # NbE    (15°)
            17: torch.tensor(( c,   -d  )),           # EbN    (75°)
            18: torch.tensor(( c,    d  )),           # EbS    (105°)
            19: torch.tensor(( d,    c  )),           # SbE    (165°)
            20: torch.tensor((-d,    c  )),           # SbW    (195°)
            21: torch.tensor((-c,    d  )),           # WbS    (255°)
            22: torch.tensor((-c,   -d  )),           # WbN    (285°)
            23: torch.tensor((-d,   -c  )),           # NbW    (345°)
        }

        if directions is None:
            #only select the first 8 directions
            self.direction_vectors = {k: v for k, v in self.direction_vectors.items() if k < 8}
        else:
            self.direction_vectors = {k: v for k, v in self.direction_vectors.items() if k in directions}
            
        self.direction_keys = list(self.direction_vectors.keys())
        # Map raw direction keys to contiguous class indices 0..N-1
        self.direction_key_to_label = {k: i for i, k in enumerate(self.direction_keys)}

        # precompute vertical bar column mask
        H, W = resolution
        self.bar_mask = None
        if num_bars > 0:
            bar_cols = torch.zeros(W, dtype=torch.bool)
            for i in range(num_bars):
                center = int((i + 0.5) * W / num_bars)
                left = max(0, center - bar_width // 2)
                right = min(W, left + bar_width)
                bar_cols[left:right] = True
            self.bar_mask = bar_cols  # shape (W,)

        # precompute (dy, dx) offsets for the chosen dot shape
        self._dot_offsets = self._compute_dot_offsets(dot_shape, dot_radius)

    def _compute_dot_offsets(self, shape, radius):
        """Return list of (dy, dx) integer offsets for rendering one dot."""
        if shape == 'point':
            return [(0, 0)]
        elif shape == 'plus':
            offsets = [(0, 0)]
            for d in range(1, radius + 1):
                offsets += [(d, 0), (-d, 0), (0, d), (0, -d)]
            return offsets
        elif shape == 'circle':
            offsets = set()
            for angle_i in range(360):
                angle = math.radians(angle_i)
                dy = round(radius * math.sin(angle))
                dx = round(radius * math.cos(angle))
                offsets.add((dy, dx))
            return list(offsets)
        else:
            raise ValueError(f"Unknown dot_shape: {shape!r}. Must be 'point', 'plus', or 'circle'.")

    def _render_dots(self, frame, xi, yi):
        H, W = self.resolution
        for dy, dx in self._dot_offsets:
            yy = torch.clamp(yi + dy, 0, H - 1)
            xx = torch.clamp(xi + dx, 0, W - 1)
            frame[yy, xx] = 1.0

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        H, W = self.resolution
        frames = []

        # pick correlation for THIS sample
        if isinstance(self.correlation, (tuple, list)):
            corr_value = torch.empty(1).uniform_(self.correlation[0], self.correlation[1]).item()
        else:
            corr_value = float(self.correlation)

        # initial positions
        x = torch.rand(self.n_dots) * W
        y = torch.rand(self.n_dots) * H

        # correlated velocity
        corr_speed = torch.randint(1, self.max_speed + 1, (1,), dtype=torch.float32)
        corr_dir_idx = torch.randint(len(self.direction_keys), (1,)).item()
        corr_dir_key = self.direction_keys[corr_dir_idx]
        corr_vec = self.direction_vectors[corr_dir_key].to(dtype=torch.float32) * corr_speed
        corr_dx, corr_dy = corr_vec[0].item(), corr_vec[1].item()

        # uncorrelated velocities (fixed per dot)
        rnd_speed = torch.rand(self.n_dots) * self.max_speed
        rnd_theta = torch.rand(self.n_dots) * (2 * math.pi)
        rnd_dx = torch.cos(rnd_theta) * rnd_speed
        rnd_dy = torch.sin(rnd_theta) * rnd_speed

        # per-sample jitter: random direction, alternates sign each frame
        if isinstance(self.jitter_speed, (tuple, list)):
            jitter_amp = torch.empty(1).uniform_(self.jitter_speed[0], self.jitter_speed[1]).item()
        else:
            jitter_amp = float(self.jitter_speed)
        if jitter_amp > 0:
            jitter_theta = torch.rand(1).item() * 2 * math.pi
            jitter_dx = math.cos(jitter_theta) * jitter_amp
            jitter_dy = math.sin(jitter_theta) * jitter_amp
        else:
            jitter_dx = jitter_dy = 0.0

        # build per-frame pulse effects: coherence multiplier and direction flip
        frame_corr_mult = [1.0] * self.n_frames
        frame_dir_flip = [False] * self.n_frames
        for start_t, duration, direction in self.evidence_pulses:
            for t in range(start_t, min(start_t + duration, self.n_frames)):
                if direction > 0:
                    frame_corr_mult[t] = 2.0
                else:
                    frame_dir_flip[t] = True

        for t in range(self.n_frames):
            frame = torch.zeros(H, W)

            # effective coherence and direction for this frame
            eff_corr = min(corr_value * frame_corr_mult[t], 1.0)
            eff_dx = -corr_dx if frame_dir_flip[t] else corr_dx
            eff_dy = -corr_dy if frame_dir_flip[t] else corr_dy

            # jitter sign alternates each frame (net displacement = 0 over even N)
            jitter_sign = 1.0 if t % 2 == 0 else -1.0

            # choose which dots are correlated this frame
            mask = torch.rand(self.n_dots) < eff_corr

            # update positions (wrap at boundaries)
            x[mask]  = torch.remainder(x[mask]  + eff_dx + jitter_sign * jitter_dx, W)
            y[mask]  = torch.remainder(y[mask]  + eff_dy + jitter_sign * jitter_dy, H)
            x[~mask] = torch.remainder(x[~mask] + rnd_dx[~mask] + jitter_sign * jitter_dx, W)
            y[~mask] = torch.remainder(y[~mask] + rnd_dy[~mask] + jitter_sign * jitter_dy, H)

            # render dots
            xi_int = torch.clamp(x.long(), 0, W - 1)
            yi_int = torch.clamp(y.long(), 0, H - 1)
            self._render_dots(frame, xi_int, yi_int)

            # zero out vertical bar columns
            if self.bar_mask is not None:
                frame[:, self.bar_mask] = 0.0

            frames.append(frame)

        frames = torch.stack(frames).unsqueeze(1)
        assert frames.shape == (self.n_frames, 1, H, W)
        label = self.direction_key_to_label[corr_dir_key]
        return frames, torch.tensor([label])
