import point_cloud_utils as pcu
import polyscope as ps
from sinkhorn import sinkhorn
import torch
from scipy.spatial.transform import Rotation
import time
import numpy as np


if __name__ == "__main__":
    torch.manual_seed(1234567)  # Fixed seed
    np.random.seed(1234567)

    niters = 500
    device = 'cuda'
    dtype = torch.float32
    eps = 1e-3
    stop_error = 1e-5
    fast_loss = False

    # Load a 3D point cloud with around 200k points and normalize it to [0, 1]^3
    print("Loading point cloud...")
    pc_load_start_time = time.time()
    p1 = torch.from_numpy(pcu.load_mesh_v("./data/wheel.ply")).to(device=device, dtype=dtype)
    p1 -= p1.min(dim=0)[0]
    p1 /= p1.max(dim=0)[0].max()
    p1 = p1[::200].contiguous()
    print(f"Done in {time.time() - pc_load_start_time}s")

    # Create a second point cloud by applying a random rotation and translation to the first one
    R = torch.from_numpy(Rotation.random().as_matrix()).to(p1)
    p2 = p1 @ R.T + 1.5
    p2.requires_grad = True

    timeframes = [p2.detach().cpu().numpy()]

    # Optimize the transformed point cloud to match the original point cloud p1 by minimizing the
    # Sinkhorn loss
    optimizer = torch.optim.Adam([p2], lr=1e-2)
    for epoch in range(101):
        optimizer.zero_grad()
        loss, corrs_1_to_2, corrs_2_to_1 = \
            sinkhorn(p1, p2, p=2, eps=eps, max_iters=niters, stop_thresh=stop_error, verbose=False)
        print(f"Sinkhorn loss is {loss.item()}")

        # For a faster approximate loss that doesn't need to backprop through Sinkhorn
        # I've found this is basically the same as using the sinkhorn loss directly
        if fast_loss:
            loss = ((p1 - p2[corrs_1_to_2]) ** 2.0).sum(-1).mean() + \
                   ((p2 - p1[corrs_2_to_1]) ** 2.0).sum(-1).mean()

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 and epoch != 0:
            timeframes.append(p2.detach().cpu().numpy())

    ps.init()
    p1 = p1.cpu().numpy()
    p2 = p2.detach().cpu().numpy()
    ps.register_point_cloud("p1", p1)
    for i in range(len(timeframes)):
        pi = timeframes[i]
        ps.register_point_cloud(f"traj_p_{i}", pi)
    ps.show()

