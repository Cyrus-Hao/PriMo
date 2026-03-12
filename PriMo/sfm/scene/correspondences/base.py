from copy import deepcopy
from itertools import chain

import numpy as np
import pycolmap

from mpsfm.baseclass import BaseClass
from mpsfm.extraction.pairwise import CONFIG_DIR as PAIRWISE_CONFIG_DIR
from mpsfm.utils.io import get_matches
from mpsfm.utils.parsers import read_unique_pairs
from mpsfm.utils.tools import load_cfg

from .utils import gather_dense_2view, gather_sparse_keypoints, gather_sparse_matches, geometric_verification


class ColmapCorrespondenceGraphWrapper:
    def __getattr__(self, name):
        return getattr(self.cg, name)


class Correspondences(BaseClass, ColmapCorrespondenceGraphWrapper):
    """MP-SfM Correspondence object. Wraps COLMAP CorrespondenceGraph and handles everything related,
    including geometric verification."""

    default_conf = {
        "matches_mode": "<--->",
        "cached_dense_scores": False,
        "verbose": 0,
    }

    def _init(self, mpsfm_rec=None, extractor=None, sfm_outputs_dir=None):
        self.cg = pycolmap.CorrespondenceGraph()
        self.mpsfm_rec = mpsfm_rec
        self.extractor = extractor
        self._two_view_geom = {}
        self.sfm_outputs_dir = sfm_outputs_dir
        self.keypoints_set = None
        self.matches_set = None
        self.sparse_im_masks = None
        self.inlier_match_scores = None

    # --- Geometry and Correspondence Access ---
    def two_view_geom(self, imname1, imname2):
        """two view geometry between image pair"""
        if (imname1, imname2) in self._two_view_geom:
            return self._two_view_geom[(imname1, imname2)], True
        if (imname2, imname1) in self._two_view_geom:
            two_view_geom = deepcopy(self._two_view_geom[(imname2, imname1)])
            two_view_geom.invert()
            return two_view_geom, True
        return None, False

    def matches(self, imid1, imid2):
        """matches between image pair"""
        return self.find_correspondences_between_images(imid1, imid2)

    def image_pairs(self):
        pairs = []
        for imid1 in self.mpsfm_rec.images:
            for imid2 in self.mpsfm_rec.images:
                if imid1 >= imid2 or self.num_correspondences_between_images(imid1, imid2) == 0:
                    continue
                pairs.append((imid1, imid2))
        return pairs

    # --- Data Import and Preparation ---
    def gather_correspondences(self, pairs, ims):
        """Gather correspondences from extractor"""
        matcher_conf = load_cfg(PAIRWISE_CONFIG_DIR / f"{self.extractor.conf.matcher}.yaml")
        if matcher_conf.type == "sparse":
            keypoints = gather_sparse_keypoints(self.extractor, ims)
            matches, scores = gather_sparse_matches(self.extractor, pairs)
            sparse_im_masks = None
        elif matcher_conf.type == "dense":
            keypoints, matches, scores, sparse_im_masks = gather_dense_2view(
                self.extractor,
                pairs,
                ims,
                matches_mode=self.conf.matches_mode,
            )
        else:
            raise ValueError(f"Unknown matcher type: {matcher_conf.type}")
        return keypoints, matches, scores, sparse_im_masks

    def gather_matches_scores(self, inlier_masks, scores, matches_set):
        """Gather match scores from inlier masks"""
        inlier_match_scores = {}
        for names, tvg in self._two_view_geom.items():
            if len(tvg.inlier_matches) == 0:
                inlier_match_scores[frozenset(names)] = 0
                continue
            if self.conf.cached_dense_scores:
                m = self.extractor.match_dirs["cache_matches"]
                _, s = get_matches(m, *names)
                if "dense" in self.conf.matches_mode and "sparse" in self.conf.matches_mode:
                    m = matches_set[names]
                    sparseA = self.sparse_im_masks[names[0]][m[:, 0]]
                    inlier_match_scores[frozenset(names)] = sum(s) if (~sparseA).sum() > 0 else 0
                else:
                    inlier_match_scores[frozenset(names)] = sum(s)
            else:
                inlm = inlier_masks[names]
                inlier_match_scores[frozenset(names)] = sum(scores[frozenset(names)][inlm])
        return inlier_match_scores

    def process_and_verify_matches(self):
        """Process correspondences and perform geometric verification"""
        pairs = read_unique_pairs(self.extractor.sfm_pairs_path)
        ims = set(chain(*pairs))
        self.keypoints_set, self.matches_set, scores, self.sparse_im_masks = self.gather_correspondences(pairs, ims)
        inlier_masks, self._two_view_geom = geometric_verification(
            self.mpsfm_rec, pairs, keypoints=self.keypoints_set, matches=self.matches_set
        )
        self.inlier_match_scores = self.gather_matches_scores(inlier_masks, scores, self.matches_set)

    # --- Graph Population ---
    def populate(self, **kwargs):
        """Initializes correspondence graph"""
        self.process_and_verify_matches()
        for im_id, image in self.mpsfm_rec.images.items():
            imname = image.name
            kps = self.keypoints_set[imname]
            self.mpsfm_rec.images[im_id].points2D = [pycolmap.Point2D(xy=kp0_i) for kp0_i in kps.astype(np.float16)]
            self.cg.add_image(im_id, kps.shape[0])
        for (name0, name1), match in self.matches_set.items():
            if match is None:
                continue
            self.cg.add_correspondences(
                self.mpsfm_rec.imid(name0),
                self.mpsfm_rec.imid(name1),
                self.two_view_geom(name0, name1)[0].inlier_matches.astype(np.uint32),
            )

        for imid in self.mpsfm_rec.images:
            if self.cg.num_correspondences_for_image(imid) == 0:
                print(f"Image {imid} has no correspondences")

        self.cg.finalize()
        return True
