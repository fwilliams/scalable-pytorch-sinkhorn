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
    stop_error = 1e-3

    # Load a 3D point cloud with around 200k points and normalize it to [0, 1]^3
    print("Loading point cloud...")
    pc_load_start_time = time.time()
    p1 = torch.from_numpy(pcu.load_mesh_v("./data/wheel.ply")).to(device=device, dtype=dtype)
    p1 -= p1.min(dim=0)[0]
    p1 /= p1.max(dim=0)[0].max()
    print(f"Done in {time.time() - pc_load_start_time}s")

    # Create a second point cloud by applying a random rotation and translation to the first one
    R = torch.from_numpy(Rotation.random().as_matrix()).to(p1)
    p2 = p1 @ R.T + 1.5

    print("Running Sinkhorn...")
    sinkhorn_start_time = time.time()
    loss, corrs_1_to_2, corrs_2_to_1 = \
        sinkhorn(p1, p2, p=2, eps=eps, max_iters=niters, stop_thresh=stop_error, verbose=True)
    torch.cuda.synchronize()
    print(f"Done in {time.time() - sinkhorn_start_time}s")

    print(f"Sinkhorn loss is {loss.item()}")

    ps.init()
    edges = torch.stack([torch.arange(p1.shape[0]).to(corrs_1_to_2),
                         corrs_1_to_2 + p1.shape[0]], dim=-1).cpu().numpy()
    verts = torch.cat([p1, p2], dim=0).cpu().numpy()
    p1 = p1.cpu().numpy()
    p2 = p2.cpu().numpy()
    ps.register_point_cloud("p1", p1)
    ps.register_point_cloud("p2", p2)
    ps.register_curve_network("corr1", verts, edges[::200]) # Only plot 100 for easier viz
    ps.show()

