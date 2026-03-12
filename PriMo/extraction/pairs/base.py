from collections.abc import Iterable
from pathlib import Path

from mpsfm.utils.io import list_h5_names
from mpsfm.utils.parsers import parse_image_lists


def pairs_from_sequential(output: Path, image_list=None, features=None, overlap=3, quadratic_overlap=True):
    """Extract sequential pairs from a list of images or features."""
    if image_list is not None:
        if isinstance(image_list, (str, Path)):
            print(image_list)
            names_q = parse_image_lists(image_list)
        elif isinstance(image_list, Iterable):
            names_q = list(image_list)
        else:
            raise ValueError(f"Unknown type for image list: {image_list}")
    elif features is not None:
        names_q = list_h5_names(features)
    else:
        raise ValueError("Provide either a list of images or a feature file.")
    pairs = []
    n = len(names_q)

    for i in range(n - 1):
        for j in range(i + 1, min(i + overlap + 1, n)):
            pairs.append((names_q[i], names_q[j]))

            if quadratic_overlap:
                q = 2 ** (j - i)
                if q > overlap and i + q < n:
                    pairs.append((names_q[i], names_q[i + q]))
    print(f"Found {len(pairs)} pairs.")
    pairs = sorted([(min(a, b), max(a, b)) for a, b in pairs])
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(" ".join([i, j]) for i, j in pairs))
    return pairs
