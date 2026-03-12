import torch
from scipy.spatial import KDTree


def sparse_nms(points, scores, nms_radius: float):
    assert points.shape[0] == scores.shape[0]

    order = torch.argsort(torch.tensor(scores), descending=True)
    points = torch.tensor(points)[order]

    tree = KDTree(points.cpu().numpy())
    keep = torch.ones(points.shape[0], dtype=torch.bool)

    for i in range(points.shape[0]):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(points[i].cpu().numpy(), nms_radius)
        neighbors = torch.tensor(neighbors, dtype=torch.long)
        keep[neighbors] = False
        keep[i] = True  # Keep the current point
    out_ids = order[keep].numpy()
    out_ids.sort()
    return out_ids
