import numpy as np
import torch
from scipy.spatial import KDTree


def simple_nms(scores, nms_radius: int):
    """Fast Non-maximum suppression to remove nearby points"""
    assert nms_radius >= 0
    assert scores.dim() == 2

    def max_pool(x):
        x = x[None, None]
        x = torch.nn.functional.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)
        return x.squeeze()

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def get_grouped_ids(array):
    idx_sort = np.argsort(array)
    sorted_array = array[idx_sort]
    _, ids, _ = np.unique(sorted_array, return_counts=True, return_index=True)
    res = np.split(idx_sort, ids[1:])
    return res


def get_unique_matches(match_ids, scores):
    if len(match_ids.shape) == 1:
        return [0]

    isets1 = get_grouped_ids(match_ids[:, 0])
    isets2 = get_grouped_ids(match_ids[:, 1])
    uid1s = [ids[scores[ids].argmax()] for ids in isets1 if len(ids) > 0]
    uid2s = [ids[scores[ids].argmax()] for ids in isets2 if len(ids) > 0]
    uids = list(set(uid1s).intersection(uid2s))
    return match_ids[uids], scores[uids]


def matches_to_matches0(matches, scores):
    if len(matches) == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float16)
    n_kps0 = np.max(matches[:, 0]) + 1
    matches0 = -np.ones((n_kps0,))
    scores0 = np.zeros((n_kps0,))
    matches0[matches[:, 0]] = matches[:, 1]
    scores0[matches[:, 0]] = scores
    return matches0.astype(np.int32), scores0.astype(np.float16)


def kpids_to_matches0(kpt_ids0, kpt_ids1, scores):
    valid = (kpt_ids0 != -1) & (kpt_ids1 != -1)
    matches = np.dstack([kpt_ids0[valid], kpt_ids1[valid]])
    matches = matches.reshape(-1, 2)
    scores = scores[valid]

    # Remove n-to-1 matches
    matches, scores = get_unique_matches(matches, scores)
    return matches_to_matches0(matches, scores)


def assign_keypoints(kpts, other_cpts, max_error):
    """Copied from hlco"""
    if len(other_cpts) == 0 or len(kpts) == 0:
        return np.full(len(kpts), -1)
    dist, kpt_ids = KDTree(np.array(other_cpts)).query(kpts, distance_upper_bound=max_error)
    kpt_ids[dist > max_error] = -1
    return kpt_ids
