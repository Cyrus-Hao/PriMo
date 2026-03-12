"""copied and adapted from hloc"""

import pprint
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

from mpsfm.data_proc import ImageDataset
from mpsfm.extraction import device, load_model
from mpsfm.utils.io import list_h5_names


@torch.no_grad()
def main(
    conf: dict,
    scene_parser,
    export_dir: Optional[Path] = None,
    as_half: bool = True,
    image_list: Optional[Union[Path, list[str]]] = None,
    feature_path: Optional[Path] = None,
    overwrite: bool = False,
    model=None,
    verbose: int = 0,
) -> Path:
    if verbose > 0:
        print("Extracting local features with configuration:" f"\n{pprint.pformat(conf)}")

    dataset = ImageDataset(scene_parser.rgb_dir, conf["preprocessing"], image_list)
    if feature_path is None:
        feature_path = Path(export_dir, conf["output"] + ".h5")
    feature_path.parent.mkdir(exist_ok=True, parents=True)
    skip_names = set(list_h5_names(feature_path) if feature_path.exists() and not overwrite else ())

    dataset.names = [n for n in dataset.names if n not in skip_names]
    if len(dataset.names) == 0:
        print("Skipping the extraction.")
        return feature_path, model
    if model is None:
        model = load_model(conf)

    loader = torch.utils.data.DataLoader(dataset, num_workers=1, shuffle=False, pin_memory=True)
    for idx, data in enumerate(tqdm(loader)):
        name = dataset.names[idx]

        pred = model({"image": data["image"].to(device, non_blocking=True)})

        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        pred["image_size"] = original_size = data["original_size"][0].numpy()
        if "keypoints" in pred:
            h, w = data["image"][0, 0].numpy().shape
            size = np.array(data["image"].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)

            pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5

            if "scales" in pred:
                pred["scales"] *= scales.mean()
            uncertainty = getattr(model, "detection_noise", 1) * scales.mean()

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        with h5py.File(str(feature_path), "a", libver="latest") as fd:
            try:
                if name in fd:
                    del fd[name]
                grp = fd.create_group(name)
                for k, v in pred.items():
                    grp.create_dataset(k, data=v)
                if "keypoints" in pred:
                    grp["keypoints"].attrs["uncertainty"] = uncertainty
            except OSError as error:
                if "No space left on device" in error.args[0]:
                    print(
                        "Out of disk space: storing features on disk can take "
                        "significant space, did you enable the as_half flag?"
                    )
                    del grp, fd[name]
                raise error

        del pred

    return Path(feature_path), model
