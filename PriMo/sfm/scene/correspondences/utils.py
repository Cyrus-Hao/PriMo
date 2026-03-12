"""Utility functions for geometric verification and gathering keypoints/matches."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pycolmap
from tqdm import tqdm

from mpsfm.utils.io import get_dense_2view_keypoints, get_keypoints, get_matches


def process_pair(data, max_error):
    """Process a pair of images to estimate two-view geometry."""
    cam0 = data["cam0"]
    kps0 = data["kps0"]
    cam1 = data["cam1"]
    kps1 = data["kps1"]
    matches = data["matches"]
    # Estimate two-view geometry
    tvg = pycolmap.estimate_calibrated_two_view_geometry(
        cam0,
        kps0,
        cam1,
        kps1,
        matches,
        {
            "ransac": {"max_num_trials": 20000, "min_inlier_ratio": 0.1, "max_error": max_error},
            "compute_relative_pose": True,
        },
    )
    return (tvg, data["matches"], data["name0"], data["name1"])


def gather_data(name0, name1, rec_name_to_id, reference, keypoints_cache, matches_cache):
    """Gather data for multi-threadedc geometric verification."""
    recid0 = rec_name_to_id[name0]
    image0 = reference.images[recid0]
    cam0 = reference.cameras[image0.camera_id].as_colmap()
    kps0 = keypoints_cache[name0]

    recid1 = rec_name_to_id[name1]
    image1 = reference.images[recid1]
    cam1 = reference.cameras[image1.camera_id].as_colmap()
    kps1 = keypoints_cache[name1]
    matches = matches_cache[name0, name1]

    return {"name0": name0, "name1": name1, "cam0": cam0, "cam1": cam1, "kps0": kps0, "kps1": kps1, "matches": matches}


def geometric_verification(
    reference: pycolmap.Reconstruction, pairs, max_error: float = 4.0, keypoints=None, matches=None
):
    """Run geometric verification on a set of image pairs."""
    tvg_cache = {}
    rec_name_to_id = {im.name: im.image_id for im in reference.images.values()}
    print("Estimating geometry and verifying matches...")
    tasks = []
    for name0, name1 in pairs:
        tasks.append(gather_data(name0, name1, rec_name_to_id, reference, keypoints, matches))

    results = []
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_pair, data, max_error) for data in tasks]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append(result)
    inlier_masks = {}
    for tvg, matches_, name0, name1 in results:
        tvg_cache[name0, name1] = tvg
        mask = np.isin(
            matches_.view([("", matches_.dtype)] * 2), tvg.inlier_matches.view([("", tvg.inlier_matches.dtype)] * 2)
        )
        inlier_masks[(name0, name1)] = mask[:, 0]

    return inlier_masks, tvg_cache


def gather_sparse_keypoints(extractor, ims):
    """Gather sparse keypoints for a set of images."""
    keypoints_cache = {}
    for name in ims:
        kps = get_keypoints(extractor.match_dirs["sfeats"], name)
        kps += 0.5  # COLMAP origin following hloc
        keypoints_cache[name] = kps.astype(np.float32)
    return keypoints_cache


def gather_sparse_matches(extractor, pairs):
    """Gather sparse matches for a set of image pairs."""
    matches_cache = {}
    scores_cache = {}
    for name0, name1 in pairs:
        matches, scores = get_matches(extractor.match_dirs["smatches"], name0, name1)
        matches_cache[name0, name1] = matches
        scores_cache[frozenset((name0, name1))] = scores
    return matches_cache, scores_cache


def gather_dense_2view(
    extractor,
    pairs,
    ims,
    matches_mode="dense",
):
    """Gather dense 2-view keypoints and matches for a set of image pairs."""
    keypoints_cache = defaultdict(list)
    matches_cache = {}
    scores_cache = {}
    sparse_im_masks = defaultdict(list)
    if "sparse" in matches_mode:
        sparse_matches = {}
        sparse_scores = {}
    if "dense" in matches_mode:
        dense_matches = {}
        dense_scores = {}

    # collect sparse
    if "sparse" in matches_mode:
        for name in ims:
            kps = get_keypoints(extractor.match_dirs["sfeats"], name)
            keypoints_cache[name].append(kps.astype(np.float32))
            sparse_im_masks[name].append(np.ones(len(kps), dtype=bool))
    for name0, name1 in pairs:
        if "sparse" in matches_mode:
            matches_primary, scores = get_matches(extractor.match_dirs["smatches"], name0, name1)
            sparse_matches[(name0, name1)] = matches_primary
            sparse_scores[frozenset((name0, name1))] = scores

    # collect dense
    for name0, name1 in pairs:
        if "dense" in matches_mode:
            matches_path = extractor.match_dirs["dmatches"]
            features_path = extractor.match_dirs["dfeats"]
            matches, scores = get_matches(matches_path, name0, name1)
            kps0, kps1 = get_dense_2view_keypoints(features_path, name0, name1)
            matches = matches[: kps0.shape[0]]

            num_keypoints0 = sum(len(kps) for kps in keypoints_cache[name0])
            num_keypoints1 = sum(len(kps) for kps in keypoints_cache[name1])
            keypoints_cache[name0].append(kps0.astype(np.float32))
            sparse_im_masks[name0].append(np.zeros(kps0.shape[0], dtype=bool))
            keypoints_cache[name1].append(kps1.astype(np.float32))
            sparse_im_masks[name1].append(np.zeros(kps1.shape[0], dtype=bool))
            matches[:, 0] += num_keypoints0
            matches[:, 1] += num_keypoints1

            dense_matches[name0, name1] = matches
            dense_scores[frozenset((name0, name1))] = scores

    # populate kps
    keypoints_cache = {k: np.concatenate(v) for k, v in keypoints_cache.items()}
    sparse_im_masks = {k: np.concatenate(v) for k, v in sparse_im_masks.items()}

    # populate matches
    for name0, name1 in pairs:
        matches = []
        scores = []
        if "dense" in matches_mode:
            matches.append(dense_matches[(name0, name1)])
            scores.append(dense_scores[frozenset((name0, name1))])
        if "sparse" in matches_mode:
            matches.append(sparse_matches[(name0, name1)])
            scores.append(sparse_scores[frozenset((name0, name1))])
        matches = np.concatenate(matches)
        if len(matches) == 0:
            matches_cache[name0, name1] = np.empty((0, 2), dtype=np.int32)
            scores_cache[frozenset((name0, name1))] = np.empty((0,), dtype=np.float32)
            continue
        matches_cache[name0, name1] = matches.astype(np.int32)
        scores_cache[frozenset((name0, name1))] = np.concatenate(scores)
    return keypoints_cache, matches_cache, scores_cache, sparse_im_masks
