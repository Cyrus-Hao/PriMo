import numpy as np
import torch

from mpsfm.extraction.pairwise.models.nearest_neighbor import NearestNeighbor  # noqa: E402


def NNs_sparse(
    pts1,
    pts2,
    scores1,
    scores2,
    kps1,
    kps2,
    scores_thresh=0.85,
    **matcher_kw,
):
    H1, W1, _ = pts1.shape
    H2, W2, _ = pts2.shape

    kps1_tensor = torch.tensor(kps1, dtype=torch.float32)  # Shape: (N, 2)
    pts1_tensor = pts1.to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    scores1_tensor = scores1.to(torch.float32)[None, None]
    grid = (
        torch.stack(
            [
                2.0 * kps1_tensor[:, 0] / (W1 - 1) - 1,  # Normalize x
                2.0 * kps1_tensor[:, 1] / (H1 - 1) - 1,  # Normalize y
            ],
            dim=-1,
        )
        .unsqueeze(0)
        .unsqueeze(2)
        .cuda()
    )
    pts1 = torch.nn.functional.grid_sample(
        pts1_tensor, grid, align_corners=True, mode="bilinear"
    )  # Shape: (1, C, N, 1)
    scores1 = torch.nn.functional.grid_sample(
        scores1_tensor, grid, align_corners=True, mode="bilinear"
    )  # Shape: (1, 1, N, 1)
    pts1 = pts1.squeeze(0).squeeze(-1).T  # Shape: (N, C)
    scores1 = scores1.squeeze()  # Shape: (N)
    kps2_tensor = torch.tensor(kps2, dtype=torch.float32)  # Shape: (N, 2)
    pts2_tensor = pts2.to(torch.float32).permute(2, 0, 1).unsqueeze(0)
    scores2_tensor = scores2.to(torch.float32)[None, None]
    grid = (
        torch.stack(
            [
                2.0 * kps2_tensor[:, 0] / (W2 - 1) - 1,  # Normalize x
                2.0 * kps2_tensor[:, 1] / (H2 - 1) - 1,  # Normalize y
            ],
            dim=-1,
        )
        .unsqueeze(0)
        .unsqueeze(2)
        .cuda()
    )
    pts2 = torch.nn.functional.grid_sample(
        pts2_tensor, grid, align_corners=True, mode="bilinear"
    )  # Shape: (1, C, N, 1)
    scores2 = torch.nn.functional.grid_sample(
        scores2_tensor, grid, align_corners=True, mode="bilinear"
    )  # Shape: (1, 1, N, 1)
    pts2 = pts2.squeeze(0).squeeze(-1).T  # Shape: (N, 24)
    scores2 = scores2.squeeze()  # Shape: (N)
    data = {"descriptors0": pts1.T.unsqueeze(0), "descriptors1": pts2.T.unsqueeze(0)}
    matcher = NearestNeighbor({"do_mutual_check": True})

    out = matcher(data)
    out["matches0"][out["matching_scores0"] < scores_thresh] = -1
    out["matches0"] = out["matches0"][0].cpu().numpy()

    matches0 = np.where(out["matches0"] != -1)[0]

    matches1 = out["matches0"][matches0]
    scores = torch.sqrt(scores1[matches0] * scores2[matches1]).cpu().numpy()
    out_scores = np.zeros(out["matches0"].shape)
    out_scores[matches0] = scores
    out["matching_scores0"] = out_scores
    return out["matches0"], out["matching_scores0"]
