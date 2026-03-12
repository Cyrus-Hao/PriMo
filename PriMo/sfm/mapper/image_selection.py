import numpy as np

from mpsfm.baseclass import BaseClass


class ImageSelection(BaseClass):
    """Class for selecting the next image to register."""

    default_conf = {
        "image_selection_method": "MAX_MATCHER_INLIER_SCORES",
        "colmap_options": "<--->",
        "verbose": 0,
    }

    def _init(self, mpsfm_rec, correspondences):
        self.mpsfm_rec = mpsfm_rec
        self.correspondences = correspondences
        self.freeze_imids = set()

        self.candid = None
        self.registration_order = []
        image_selection_methods = {
            "MAX_VISIBLE_POINTS_NUM": self.rank_next_image_max_visible_points_num,
            "MAX_VISIBLE_POINTS_RATIO": self.rank_next_image_max_visible_points_ratio,
            "MIN_UNCERTAINTY": self.rank_next_image_min_uncertainty,
            "MAX_NUM_CORRESPONDENCES": self.rank_next_image_max_num_correspondences,
            "MAX_NUM_INLIER_CORRESPONDENCES": self.rank_next_image_max_num_inlier_correspondences,
            "MAX_NUM_INLIER_CORRESPONDENCES_TOT": self.rank_next_image_max_num_inlier_correspondences_tot,
            "MAX_NUM_INLIER_SCORES_TOT": self.rank_next_image_max_num_inlier_matcher_scores_tot,
            "MAX_MATCHER_INLIER_SCORES": self.rank_next_image_max_sum_inlier_matcher_scores,
        }
        method_key = self.conf.image_selection_method
        if method_key not in image_selection_methods:
            raise ValueError(f"Image selection method {method_key} not recognized")
        self.rank_image_func = image_selection_methods[method_key]

    def selection_method(self, num_inliers, **kwargs):
        num_inliers = np.array(num_inliers)
        rank_ids = np.argsort(num_inliers)[::-1]
        return rank_ids

    def find_init_pairs(self, exclude_init_pairs=None):
        """Finds the best pair of images to initialize the reconstruction."""
        two_view_config = 2
        propose_pairs = []
        for j in range(7):
            two_view_config = 2 + j
            if exclude_init_pairs is None:
                exclude_init_pairs = []
            num_inliers = []
            tri_angles = []
            impairs = [
                pair
                for pair in self.mpsfm_rec.filtered_image_pairs(self.correspondences.two_view_geom, two_view_config)
                if pair not in exclude_init_pairs
            ]
            if len(impairs) == 0:
                continue
            for imid1, imid2 in impairs:
                two_view_geom, success = self.correspondences.two_view_geom(
                    self.mpsfm_rec.images[imid1].name, self.mpsfm_rec.images[imid2].name
                )
                if not success:
                    num_inliers.append(1e-6)
                    tri_angles.append(1e-6)
                    continue

                num_inliers.append(two_view_geom.inlier_matches.shape[0])
                tri_angles.append(two_view_geom.tri_angle)

            valid_init_pairs = self.selection_method(
                num_inliers=num_inliers,
                tri_angles=tri_angles,
            )
            propose_pairs.append([impairs[i] for i in valid_init_pairs])
        return sum(propose_pairs, [])

    def rank_next_image_max_visible_points_num(self, imid):
        """Rank the next image based on the maximum number of visible points."""
        return {"score": self.mpsfm_rec.obs.num_visible_points3D(imid)}

    def rank_next_image_max_visible_points_ratio(self, imid):
        """Rank the next image based on the maximum ratio of visible points."""
        return {"score": self.mpsfm_rec.obs.num_visible_points3D(imid) / self.mpsfm_rec.obs.num_observations(imid)}

    def rank_next_image_min_uncertainty(self, imid):
        """Default COLMAP approach."""
        return {"score": self.mpsfm_rec.obs.point3D_visibility_score(imid)}

    def rank_next_image_sum_num_correspondences(self, imid):
        """Rank the next image based on the sum of correspondences between qry and entire map."""
        num_corresp = self.correspondences.num_correspondences_between_images
        reg_imids = self.mpsfm_rec.registered_images.keys()
        return {"score": sum(num_corresp(imid, reg_imid) for reg_imid in reg_imids)}

    def rank_next_image_max_num_correspondences(self, imid):
        """Rank the next image based on the maximum number of correspondences."""
        num_corresp = self.correspondences.num_correspondences_between_images
        reg_imids = list(self.mpsfm_rec.registered_images.keys())
        qry_scores = [num_corresp(imid, reg_imid) for reg_imid in reg_imids]
        amax = np.argmax(qry_scores)
        return {"score": qry_scores[amax], "refid": reg_imids[amax]}

    def rank_next_image_max_num_inlier_correspondences(self, imid):
        """Rank the next image based on the maximum number of inlier correspondences."""
        rec = self.mpsfm_rec
        reg_ims = list(rec.registered_images.values())
        tvg_out = [self.correspondences.two_view_geom(rec.images[imid].name, ref_im.name) for ref_im in reg_ims]
        tvg_inlier_count = [tvg.inlier_matches.shape[0] if valid else 0 for tvg, valid in tvg_out]
        amax = np.argmax(tvg_inlier_count)
        return {"score": tvg_inlier_count[amax], "refid": reg_ims[amax].image_id}

    def rank_next_image_max_num_inlier_correspondences_tot(self, imid):
        """Rank the next image based on the maximum number of inlier correspondences between qry and entire map."""
        rec = self.mpsfm_rec
        reg_ims = list(rec.registered_images.values())
        tvg_out = [self.correspondences.two_view_geom(rec.images[imid].name, ref_im.name) for ref_im in reg_ims]
        tvg_inlier_count = [tvg.inlier_matches.shape[0] if valid else 0 for tvg, valid in tvg_out]
        amax = np.argmax(tvg_inlier_count)
        return {"score": np.sum(tvg_inlier_count), "refid": reg_ims[amax].image_id}

    def rank_next_image_max_num_inlier_matcher_scores_tot(self, imid):
        """Rank the next image based on the maximum number of inlier correspondence scores
        between qry and entire map."""
        rec = self.mpsfm_rec
        reg_ims = list(rec.registered_images.values())
        qry_scores = [
            self.correspondences.inlier_match_scores.get(frozenset([rec.images[imid].name, ref_im.name]), 0)
            for ref_im in reg_ims
        ]
        amax = np.argmax(qry_scores)
        return {"score": np.sum(qry_scores), "refid": reg_ims[amax].image_id}

    def rank_next_image_max_sum_inlier_matcher_scores(self, imid):
        """Rank the next image based on the maximum sum of inlier correspondence scores."""
        rec = self.mpsfm_rec
        reg_ims = list(rec.registered_images.values())

        qry_scores = [
            self.correspondences.inlier_match_scores.get(frozenset([rec.images[imid].name, ref_im.name]), 0)
            for ref_im in reg_ims
        ]
        for ii, reg_im in enumerate(reg_ims):
            if reg_im.imid in self.mpsfm_rec.images[imid].ignore_matches_AP:
                ignore_mask = self.mpsfm_rec.images[imid].ignore_matches_AP[reg_im.imid]
                qry_scores[ii] *= (~ignore_mask).sum() / ignore_mask.sum()
        amax = np.argmax(qry_scores)
        return {"score": qry_scores[amax], "refid": reg_ims[amax].image_id}

    def next_image(self, qry_imids=None):
        """Select the next image to register based on the ranking function."""
        if qry_imids is None:
            qry_imids = [
                im_id
                for im_id, image in self.mpsfm_rec.images.items()
                if not image.has_pose and im_id not in self.freeze_imids
            ]
        if len(qry_imids) == 0:
            return False

        rank_fn_out = [self.rank_image_func(qry_id) for qry_id in qry_imids]
        scores = [out["score"] for out in rank_fn_out]
        best_id = np.argsort(scores)[-1]
        self.mpsfm_rec.best_next_ref_imid = rank_fn_out[best_id].get("refid")
        self.candid = qry_imids[best_id]
        return True

    def at_success(self):
        """Update the registration order after a successful registration and reset frozen images."""
        self.freeze_imids = set()
        self.registration_order.append(self.candid)
        if self.conf.verbose > 1:
            print(50 * "=")
            print(f"Image {self.mpsfm_rec.images[self.candid].name} registered")
            print(50 * "=")

    def at_failure(self, imid):
        """Update the registration order after a failed registration and freeze the image."""
        self.freeze_imids.add(imid)
