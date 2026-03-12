"""copied and adapted from hloc"""

from contextlib import ExitStack
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

from mpsfm.data_proc import ImagePairDataset
from mpsfm.extraction import device, load_model
from mpsfm.extraction.pairwise.match_sparse import find_unique_new_pairs
from mpsfm.utils.parsers import names_to_pair, parse_retrieval

from .models.utils.generic import sparse_nms


def scale_keypoints(kpts, scale):
    if np.any(scale != 1.0):
        kpts *= kpts.new_tensor(scale)
    return kpts


@torch.no_grad()
def match_dense(
    conf: dict,
    pairs: list[tuple[str, str]],
    image_dir: Path,
    match_dirs: dict,
    matches_mode=None,
    model=None,
    sparse_nms_radius=None,
):
    """Main function to match sparse features and extract dense correspondences with dense matcher."""
    if model is None:
        model = load_model(conf)
    dataset = ImagePairDataset(image_dir, conf.get("preprocessing", {}), pairs)
    loader = torch.utils.data.DataLoader(dataset, num_workers=16, batch_size=1, shuffle=False)

    file_args = [
        ("sparse", match_dirs.get("sfeats"), "r"),
        ("sparse", match_dirs.get("smatches"), "a"),
        ("dense", match_dirs.get("dfeats"), "a"),
        ("dense", match_dirs.get("dmatches"), "a"),
        ("cache", match_dirs.get("cache_dfeats"), "a"),
        ("cache", match_dirs.get("cache_matches"), "a"),
        ("depth", match_dirs.get("depth"), "a"),
    ]

    print("Performing dense matching...")
    with ExitStack() as stack:

        def open_h5(name, path, mode):
            return stack.enter_context(h5py.File(str(path), mode)) if (name in matches_mode) else None

        fd_sfeats, fd_smatches, fd_dfeats, fd_dmatches, fd_cfeats, fd_cmatches, fd_depths = [
            open_h5(*args) for args in file_args
        ]

        for data in tqdm(loader, smoothing=0.1):
            image0, image1, scale0, scale1, (name0,), (name1,) = data
            scale0, scale1 = scale0[0].numpy(), scale1[0].numpy()
            image0, image1 = image0.to(device), image1.to(device)
            input_data = {
                "image0": image0,
                "image1": image1,
                "name0": name0,
                "name1": name1,
                "imdir": image_dir,
                "scale0": scale0,
                "scale1": scale1,
            }
            if "sparse" in matches_mode:
                input_data["skpts0"] = fd_sfeats[name0]["keypoints"][()]
                input_data["skpts1"] = fd_sfeats[name1]["keypoints"][()]

            mode = matches_mode
            if "cache" in matches_mode:
                mode += "+dense"
            pred = model(input_data, mode=mode)

            pair = names_to_pair(name0, name1)
            if "sparse" in matches_mode:
                if pair in fd_smatches:
                    del fd_smatches[pair]
                smgrp = fd_smatches.create_group(pair)
                smgrp.create_dataset("matches0", data=pred["smatches0"])
                smgrp.create_dataset("matching_scores0", data=pred["smatching_scores0"])
                if names_to_pair(name1, name0) in smgrp:
                    del smgrp[names_to_pair(name1, name0)]

            if "dense" in matches_mode or "cache" in matches_mode:
                dkpts0 = pred["dkeypoints0"]
                dkpts1 = pred["dkeypoints1"]
                dkpts0 = torch.Tensor(dkpts0)
                dkpts1 = torch.Tensor(dkpts1)
                dscores = torch.Tensor(pred["dscores"])
                if scale0[0] != 1 or scale0[1] != 1:
                    dkpts0 = scale_keypoints(dkpts0 + 0.5, scale0) - 0.5
                if scale1[0] != 1 or scale1[1] != 1:
                    dkpts1 = scale_keypoints(dkpts1 + 0.5, scale1) - 0.5

                if "cache" in matches_mode:
                    cache_dkpts0 = dkpts0.cpu().numpy()
                    cache_dkpts1 = dkpts1.cpu().numpy()
                    cache_dscores = dscores.cpu().numpy()
                    if pair in fd_cmatches:
                        del fd_cmatches[pair]
                    if f"{pair}/{name0}" in fd_cfeats:
                        del fd_cfeats[f"{pair}/{name0}"]
                    if f"{pair}/{name1}" in fd_cfeats:
                        del fd_cfeats[f"{pair}/{name1}"]
                    cdmgrp = fd_cmatches.create_group(pair)
                    ckgrp0 = fd_cfeats.create_group(f"{pair}/{name0}")
                    ckgrp1 = fd_cfeats.create_group(f"{pair}/{name1}")

                    ckgrp0.create_dataset("keypoints", data=cache_dkpts0)
                    ckgrp1.create_dataset("keypoints", data=cache_dkpts1)
                    ckgrp0.create_dataset("scores", data=cache_dscores)
                    ckgrp1.create_dataset("scores", data=cache_dscores)
                    cmatches0 = np.arange(len(cache_dkpts0), dtype=np.int32)
                    cdmgrp.create_dataset("matches0", data=cmatches0)
                    cdmgrp.create_dataset("matching_scores0", data=cache_dscores)

                if "dense" in matches_mode:
                    if "sparse" in matches_mode:
                        smatches0 = np.where(pred["smatches0"] != -1)[0]
                        smatches1 = pred["smatches0"][smatches0]
                        skpts0_matched = input_data["skpts0"][smatches0]
                        skpts1_matched = input_data["skpts1"][smatches1]
                        kps0_comb = np.concatenate([skpts0_matched, dkpts0])
                        scores_comb = np.concatenate([np.ones(skpts0_matched.shape[0]) * 100, dscores], axis=0)
                        mask = (
                            sparse_nms(kps0_comb, scores_comb, sparse_nms_radius)[skpts0_matched.shape[0] :]
                            - skpts0_matched.shape[0]
                        )
                        dkpts0 = dkpts0[mask]
                        dkpts1 = dkpts1[mask]
                        dscores = dscores[mask]
                        kps1_comb = np.concatenate([skpts1_matched, dkpts1])

                        scores_comb = np.concatenate([np.ones(skpts1_matched.shape[0]) * 100, dscores], axis=0)
                        mask = (
                            sparse_nms(kps1_comb, scores_comb, sparse_nms_radius)[skpts1_matched.shape[0] :]
                            - skpts1_matched.shape[0]
                        )
                        dkpts0 = dkpts0[mask]
                        dkpts1 = dkpts1[mask]
                        dscores = dscores[mask]
                    else:
                        # if not sparse we still optionally still subsample
                        mask = sparse_nms(dkpts0, dscores, sparse_nms_radius)
                        dkpts0 = dkpts0[mask]
                        dkpts1 = dkpts1[mask]
                        dscores = dscores[mask]
                        mask = sparse_nms(dkpts1, dscores, sparse_nms_radius)
                        dkpts0 = dkpts0[mask]
                        dkpts1 = dkpts1[mask]
                        dscores = dscores[mask]

                    if isinstance(dkpts0, torch.Tensor):
                        dkpts0 = dkpts0.cpu().numpy()
                    if isinstance(dkpts1, torch.Tensor):
                        dkpts1 = dkpts1.cpu().numpy()
                    if isinstance(dscores, torch.Tensor):
                        dscores = dscores.cpu().numpy()

                    if pair in fd_dmatches:
                        del fd_dmatches[pair]
                    if f"{pair}/{name0}" in fd_dfeats:
                        del fd_dfeats[f"{pair}/{name0}"]
                    if f"{pair}/{name1}" in fd_dfeats:
                        del fd_dfeats[f"{pair}/{name1}"]
                    dmgrp = fd_dmatches.create_group(pair)
                    kgrp0 = fd_dfeats.create_group(f"{pair}/{name0}")
                    kgrp1 = fd_dfeats.create_group(f"{pair}/{name1}")

                    # Write dense matching output
                    kgrp0.create_dataset("keypoints", data=dkpts0)
                    kgrp1.create_dataset("keypoints", data=dkpts1)
                    kgrp0.create_dataset("scores", data=dscores)
                    kgrp1.create_dataset("scores", data=dscores)
                    matches0 = np.arange(len(dkpts0), dtype=np.int32)
                    scores0 = dscores
                    dmgrp.create_dataset("matches0", data=matches0)
                    dmgrp.create_dataset("matching_scores0", data=scores0)
                    if names_to_pair(name1, name0) in dmgrp:
                        del dmgrp[names_to_pair(name1, name0)]
            if "depth" in matches_mode:
                if f"{pair}/{name0}" in fd_depths:
                    del fd_depths[f"{pair}/{name0}"]
                if f"{pair}/{name1}" in fd_depths:
                    del fd_depths[f"{pair}/{name1}"]
                dgrp0 = fd_depths.create_group(f"{pair}/{name0}")
                dgrp1 = fd_depths.create_group(f"{pair}/{name1}")
                dgrp0.create_dataset("depth", data=pred["depth0"])
                dgrp1.create_dataset("depth", data=pred["depth1"])
                dgrp0.create_dataset("valid", data=pred["valid0"])
                dgrp1.create_dataset("valid", data=pred["valid1"])
                if "variance0" in pred:
                    dgrp0.create_dataset("variance", data=pred["variance0"])
                    dgrp1.create_dataset("variance", data=pred["variance1"])

    del loader
    return model


@torch.no_grad()
def match_and_assign(
    conf: dict,
    pairs_path: Path,
    image_dir: Path,
    match_dirs: dict,
    overwrite: bool = False,
    model=None,
    matches_mode=None,
    sparse_nms_radius=None,
    verbose: int = 0,
) -> Path:
    pairs = parse_retrieval(pairs_path)
    pairs = [tuple(sorted((q, r))) for q, rs in pairs.items() for r in rs]
    extract_pairs = set()
    if "sparse" in matches_mode:
        new_sparse_pairs = set(find_unique_new_pairs(pairs, None if overwrite else match_dirs["smatches"]))
        extract_pairs |= new_sparse_pairs
        print(f"Extracting {len(new_sparse_pairs)} new sparse pairs")
    if "dense" in matches_mode:
        new_dense_pairs = set(find_unique_new_pairs(pairs, None if overwrite else match_dirs["dmatches"]))
        extract_pairs |= new_dense_pairs
        print(f"Extracting {len(new_dense_pairs)} new dense pairs")
    if "cache" in matches_mode:
        new_cache_pairs = set(find_unique_new_pairs(pairs, None if overwrite else match_dirs["cache_matches"]))
        extract_pairs |= new_cache_pairs
        print(f"Extracting {len(new_cache_pairs)} new cache pairs")
    if "depth" in matches_mode:
        new_depth_pairs = set(find_unique_new_pairs(pairs, None if overwrite else match_dirs["depth"]))
        extract_pairs |= new_depth_pairs
        print(f"Extracting {len(new_depth_pairs)} new depth pairs")
    pairs = list(extract_pairs)
    if verbose > 0:
        print(f"Extracting {len(pairs)} pairs")
    if len(pairs) == 0:
        print("Skipping dense matching.")
        return

    model = match_dense(
        conf,
        pairs,
        image_dir,
        match_dirs,
        model=model,
        matches_mode=matches_mode,
        sparse_nms_radius=sparse_nms_radius,
    )
    print("Assigning matches...")
    return model


@torch.no_grad()
def main(
    conf: dict,
    pairs: Path,
    scene_parser,
    match_dirs: dict,
    export_dir: Optional[Path] = None,
    overwrite: bool = False,
    model=None,
    matches_mode=None,
    sparse_nms_radius=None,
    verbose: int = 0,
) -> Path:
    assert export_dir is not None, "Export directory is not set."
    assert matches_mode is not None, "Matches mode is not set."
    assert "sparse" in matches_mode or "dense" in matches_mode, "Matches mode should be 'sparse' or 'dense'."
    if "sparse" in matches_mode:
        assert "sfeats" in match_dirs, "Sparse features are not set."
        match_dirs["smatches"] = Path(export_dir, f"smatches-{match_dirs['sfeats'].stem}_{conf['output']}.h5")
        if verbose:
            print("Input sparse features:", match_dirs["sfeats"])
            print("Output sparse matches:", match_dirs["smatches"])

    if "dense" in matches_mode:
        # if we use sparse matches, dense match locations can be different
        base_name = f"{match_dirs['sfeats'].stem}_" if ("sparse" in matches_mode) else ""
        match_dirs["dfeats"] = Path(export_dir, f'dfeats-{base_name}{conf["output"]}.h5')
        match_dirs["dmatches"] = Path(export_dir, f'dmatches_{base_name}{conf["output"]}.h5')
        if verbose:
            print("Output dense features:", match_dirs["dfeats"])
            print("Output dense matches:", match_dirs["dmatches"])

    if "cache" in matches_mode:
        # dense matches will be supsampled boying the network output. option to cache semi-dense
        base_name = f"{match_dirs['sfeats'].stem}_"
        match_dirs["cache_dfeats"] = Path(export_dir, f'cache_dfeats-{base_name}{conf["output"]}.h5')
        match_dirs["cache_matches"] = Path(export_dir, f'cache_dmatches_{base_name}{conf["output"]}.h5')
        print("Output cache dense features:", match_dirs["cache_dfeats"])
        print("Output cache dense matches:", match_dirs["cache_matches"])

    if "depth" in matches_mode:
        # depths do not depend on whether we use sparse or dense matches
        match_dirs["depth"] = Path(export_dir, f'depths-{conf["output"]}.h5')
        if verbose:
            print("Output depth matches:", match_dirs["depth"])

    model = match_and_assign(
        conf,
        pairs,
        scene_parser.rgb_dir,
        match_dirs,
        overwrite,
        model=model,
        matches_mode=matches_mode,
        sparse_nms_radius=sparse_nms_radius,
        verbose=verbose,
    )
    return match_dirs, model
