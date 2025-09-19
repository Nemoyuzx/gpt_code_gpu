import random
import numpy as np
import math

def make_batch_func_factory(sim):
    # 从 walls 估计一个采样边界盒
    xs = [p[0] for seg in sim.walls for p in seg]
    ys = [p[1] for seg in sim.walls for p in seg]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    def sample_pose():
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        theta = random.uniform(-math.pi, math.pi)
        return (x, y, theta)

    num_beams = int(360 / sim.angle_resolution)

    def make_batch(batch_size):
        noisy_list, clean_list = [], []
        for _ in range(batch_size):
            pose = sample_pose()
            noisy, clean = sim.scan_with_clean(pose)
            # 转 np.array 便于后续 DataLoader
            noisy = np.asarray(noisy, dtype=np.float32)
            clean = np.asarray(clean, dtype=np.float32)
            assert noisy.shape[0] == num_beams and clean.shape[0] == num_beams
            noisy_list.append(noisy)
            clean_list.append(clean)
        return noisy_list, clean_list

    return make_batch
