"""I/O functions for reading and writing data. Many copied from and inspired by hloc."""

from pathlib import Path

import cv2
import h5py
import numpy as np

from mpsfm.utils.parsers import names_to_pair, names_to_pair_old, read_unique_pairs


def get_mono_map(path, name):
    with h5py.File(str(path), "r") as f:
        return {k: v[:] for k, v in f[str(Path(name).name)].items()}


def get_mono_map_from_pairs(path, name, pairs_path):
    pairs = read_unique_pairs(pairs_path)
    cname = str(Path(name).name)
    with h5py.File(str(path), "r") as f:
        depths = []
        valids = []
        variances = []
        scores = []
        for pair in pairs:
            if cname not in pair:
                continue
            key = f"{names_to_pair(*pair)}/{cname}"
            depths.append(f[key]["depth"][:])
            valids.append(f[key]["valid"][:])
            variances.append(f[key]["variance"][:])
            scores.append(((1 / variances[-1])[valids[-1]]).mean())

        if len(scores) == 0:
            return None
        amax = np.argmax(scores)

        return {
            "depth": depths[amax],
            "valid": valids[amax],
            "depth_variance": variances[amax],
        }


def get_mask(path, name):
    with h5py.File(path, "r") as file:
        return file[name]["mask"][:]


def read_image(path, grayscale=False):
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


def list_h5_names(path):
    names = []
    with h5py.File(str(path), "r", libver="latest") as fd:

        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip("/"))

        fd.visititems(visit_fn)
    return list(set(names))


def get_keypoints(path: Path, name: str, return_uncertainty: bool = False) -> np.ndarray:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        dset = hfile[name]["keypoints"]
        p = dset.__array__()
        uncertainty = dset.attrs.get("uncertainty")
    if return_uncertainty:
        return p, uncertainty
    return p


def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(f"Could not find pair {(name0, name1)}... " "Maybe you matched with a different list of pairs? ")


def get_dense_2view_keypoints(path: Path, name0: str, name1: str):
    with h5py.File(path, "r", libver="latest") as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        kps0 = hfile[pair][name0]["keypoints"].__array__()
        kps1 = hfile[pair][name1]["keypoints"].__array__()
    return kps0, kps1


def get_matches(path: Path, name0: str, name1: str) -> tuple[np.ndarray]:
    with h5py.File(str(path), "r", libver="latest") as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]["matches0"].__array__()
        scores = hfile[pair]["matching_scores0"].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    return matches, scores


def write_match_correspondence_counts(match_path: Path, output_path: Path) -> bool:
    """Write per-pair correspondence counts from a matches .h5 file."""
    if match_path is None:
        return False
    match_path = Path(match_path)
    if not match_path.exists():
        return False

    per_pair = []
    total_pairs = 0
    total_correspondences = 0
    missing_matches0 = 0

    with h5py.File(str(match_path), "r", libver="latest") as hfile:
        # Two common layouts:
        # 1) /pair_name/{matches0, matching_scores0}
        # 2) /name0/name1/{matches0, matching_scores0}
        has_flat = any(
            isinstance(hfile[key], h5py.Group) and "matches0" in hfile[key] for key in hfile.keys()
        )
        if has_flat:
            for pair in hfile:
                grp = hfile[pair]
                if "matches0" not in grp:
                    missing_matches0 += 1
                    continue
                matches0 = grp["matches0"][()]
                count = int(np.count_nonzero(matches0 != -1))
                per_pair.append((pair, count))
                total_pairs += 1
                total_correspondences += count
        else:
            for name0 in hfile:
                grp0 = hfile[name0]
                for name1 in grp0:
                    grp1 = grp0[name1]
                    if not isinstance(grp1, h5py.Group) or "matches0" not in grp1:
                        missing_matches0 += 1
                        continue
                    matches0 = grp1["matches0"][()]
                    count = int(np.count_nonzero(matches0 != -1))
                    pair = f"{name0}/{name1}"
                    per_pair.append((pair, count))
                    total_pairs += 1
                    total_correspondences += count

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"matches_path: {match_path}\n")
        f.write(f"total_pairs: {total_pairs}\n")
        f.write(f"total_correspondences: {total_correspondences}\n")
        if missing_matches0:
            f.write(f"pairs_missing_matches0: {missing_matches0}\n")
        f.write("pair correspondence_count\n")
        for pair, count in sorted(per_pair, key=lambda x: x[0]):
            f.write(f"{pair} {count}\n")
    return True
