import cv2
import h5py
import numpy as np
from tqdm import tqdm

from mpsfm.data_proc import get_dataset
from mpsfm.extraction import load_model
from mpsfm.utils.io import list_h5_names


def write(pred, output_path):
    name = pred.pop("name")
    with h5py.File(str(output_path), "a", libver="latest") as fd:
        if name in fd:
            del fd[name]
        grp = fd.create_group(name)
        for k, v in pred.items():
            grp.create_dataset(k, data=v)


def extract(data, model, rot_images=False, skip_names=None):
    if skip_names is None:
        skip_names = {}
    input_data = {}
    name = data["meta"]["image_name"]

    assert len(name) == 1
    name = name[0]

    if name in skip_names:
        return None
    scale = model.conf.scale if hasattr(model.conf, "scale") else 1
    image = (data["image"].numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
    if scale != 1:
        image = cv2.resize(
            image,
            None,
            fx=model.conf.scale,
            fy=model.conf.scale,
            interpolation=cv2.INTER_AREA,
        )
    input_data["image"] = image
    input_data["meta"] = data["meta"]

    pred = model(input_data)
    pred["name"] = name
    return pred


def main(conf, export_dir, overwrite=False, image_list=None, model=None, scene_parser=None, verbose=0):
    export_dir.mkdir(parents=True, exist_ok=True)
    write_name = conf.model.write_name if "write_name" in conf.model else conf.model.name
    output_path = export_dir / f"{write_name}.h5"
    skip_names = set(list_h5_names(output_path) if output_path.exists() and not overwrite else ())
    extract_num = len(image_list)
    image_list = [f for f in image_list if f not in skip_names]
    if verbose > 0:
        print(f"Skipping {extract_num-len(image_list)} files")
    loader = get_dataset(conf.dataset.name)(
        conf.dataset, image_list=image_list, scene_parser=scene_parser, cache_dir=export_dir
    ).get_dataloader()
    if verbose > 0 and len(skip_names) > 0:
        print(f"Skipping {len(skip_names)} files")
    if len(loader) == 0:
        print("No files to extract")
        return output_path, model
    if model is None:
        model = load_model(conf)

    for data in tqdm(loader):
        pred = extract(data, model, skip_names)
        if pred is None:
            continue
        write(pred, output_path)
    return output_path, model
