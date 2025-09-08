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
    ):
        """
        Args:
            n_frames (int): Number of frames to generate.
            n_dots (int): Number of dots in each frame.
            resolution (tuple): Size of each frame (height, width).
            correlation (float or tuple): If float, fixed correlation for all samples.
                If tuple (low, high), each sample’s correlation is drawn uniformly
                from [low, high].
            max_speed (int): Max pixels per frame.
        """
        self.n_frames = n_frames
        self.n_dots = n_dots
        self.resolution = resolution
        self.correlation = correlation
        self.max_speed = max_speed
        self.samples_per_epoch = samples_per_epoch

        # 8 directions: R, L, U, D, NE, NW, SE, SW
        r2 = math.sqrt(2.0)
        self.direction_vectors = {
            0: torch.tensor(( 1.0,  0.0)),         # Right
            1: torch.tensor((-1.0,  0.0)),         # Left
            2: torch.tensor(( 0.0, -1.0)),         # Up
            3: torch.tensor(( 0.0,  1.0)),         # Down
            4: torch.tensor(( 1.0, -1.0)) / r2,    # NE
            5: torch.tensor((-1.0, -1.0)) / r2,    # NW
            6: torch.tensor(( 1.0,  1.0)) / r2,    # SE
            7: torch.tensor((-1.0,  1.0)) / r2,    # SW
        }

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
        corr_dir = torch.randint(len(self.direction_vectors), (1,))
        corr_vec = self.direction_vectors[corr_dir.item()].to(dtype=torch.float32) * corr_speed
        corr_dx, corr_dy = corr_vec[0].item(), corr_vec[1].item()

        # uncorrelated velocities (fixed per dot)
        rnd_speed = torch.rand(self.n_dots) * self.max_speed
        rnd_theta = torch.rand(self.n_dots) * (2 * torch.pi)
        rnd_dx = torch.cos(rnd_theta) * rnd_speed
        rnd_dy = torch.sin(rnd_theta) * rnd_speed

        for _ in range(self.n_frames):
            frame = torch.zeros(H, W)

            # choose which dots are correlated this frame
            mask = torch.rand(self.n_dots) < corr_value

            # update positions
            x[mask]  = torch.remainder(x[mask]  + corr_dx, W)
            y[mask]  = torch.remainder(y[mask]  + corr_dy, H)
            x[~mask] = torch.remainder(x[~mask] + rnd_dx[~mask], W)
            y[~mask] = torch.remainder(y[~mask] + rnd_dy[~mask], H)

            # safe integer indices
            xi = torch.clamp(x.long(), 0, W - 1)
            yi = torch.clamp(y.long(), 0, H - 1)
            frame[yi, xi] = 1.0

            frames.append(frame)

        frames = torch.stack(frames).unsqueeze(1)
        assert frames.shape == (self.n_frames, 1, H, W)
        return frames, corr_dir
