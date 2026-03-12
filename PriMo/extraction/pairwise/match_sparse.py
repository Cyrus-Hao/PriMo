"""copied and adapted from hloc"""

import pprint
from functools import partial
from pathlib import Path
from typing import Optional, Union

import h5py
import torch
from tqdm import tqdm

from mpsfm.data_proc import FeaturePairsDataset, WorkQueue, writer_fn
from mpsfm.extraction import device, load_model
from mpsfm.utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval


def main(
    conf: dict,
    pairs: Path,
    features: Union[Path, str],
    export_dir: Optional[Path] = None,
    matches: Optional[Path] = None,
    features_ref: Optional[Path] = None,
    overwrite: bool = False,
    rot_images=None,
    model=None,
    verbose: int = 0,
) -> Path:
    """Main function to match sparse features."""
    if verbose > 0:
        print("Matching local features with configuration:" f"\n{pprint.pformat(conf)}")
    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError("Either provide both features and matches as Path or both as names.")
    else:
        if export_dir is None:
            raise ValueError("Provide an export_dir if features is not" f" a file path: {features}.")
        features_q = Path(export_dir, features + ".h5")
        if matches is None:
            matches = Path(export_dir, f'{features}_{conf["output"]}_pairs.h5')

    if features_ref is None:
        features_ref = features_q
    model = match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite, rot_images, model=model)

    return Path(matches), model


def find_unique_new_pairs(pairs_all: list[tuple[str]], match_path: Path = None):
    """Avoid to recompute duplicates to save time."""
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), "r", libver="latest") as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (
                    names_to_pair(i, j) in fd
                    or names_to_pair(j, i) in fd
                    or names_to_pair_old(i, j) in fd
                    or names_to_pair_old(j, i) in fd
                ):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(
    conf: dict,
    pairs_path: Path,
    match_path: Path,
    feature_path_q: Path,
    feature_path_ref: Path,
    overwrite: bool = False,
    rot_images=None,
    model=None,
) -> Path:

    if not feature_path_q.exists():
        raise FileNotFoundError(f"Query feature file {feature_path_q}.")
    if not feature_path_ref.exists():
        raise FileNotFoundError(f"Reference feature file {feature_path_ref}.")
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        print("Skipping the matching.")
        return None

    if model is None:
        model = load_model(conf)

    dataset = FeaturePairsDataset(pairs, feature_path_q, feature_path_ref)
    loader = torch.utils.data.DataLoader(dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True)
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=0.1)):
        pair = names_to_pair(*pairs[idx])
        imname0, imname1 = pair.split("/")

        data = {k: v if k.startswith("image") else v.to(device, non_blocking=True) for k, v in data.items()}

        if rot_images is not None:

            def adjust_keypoints_array(keypoints, imshape, n_rotations_clockwise):
                _, _, h, w = imshape  # Original image dimensions
                for _ in range(n_rotations_clockwise):
                    keypoints = torch.column_stack((w - 1 - keypoints[0, :, 1], keypoints[0, :, 0]))[None]
                    h, w = h, w
                return keypoints

            data["keypoints0"] = adjust_keypoints_array(
                data["keypoints0"], data["image0"].shape, (4 - rot_images[imname0]) % 4
            )
            data["keypoints1"] = adjust_keypoints_array(
                data["keypoints1"], data["image1"].shape, (4 - rot_images[imname1]) % 4
            )
        pred = model(data)

        writer_queue.put((pair, pred))
    writer_queue.join()
    print("Finished exporting matches.")
    return model
