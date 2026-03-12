import numpy as np

from mpsfm.baseclass import BaseClass


class DepthConsistencyChecker(BaseClass):
    """Class to check depth consistency of an image."""

    default_conf = {
        "depth_cons_valid_thresh": 0.6,
        "depth_cons_thresh": 0.15,
        "init_depth_cons_thresh": 0.09,
        "init_valid_thresh": 0.8,
        "depth_consistency_resample": False,  # exploration
        "verbose": 0,
    }

    def _init(self, mpsfm_rec, correspondences):
        self.mpsfm_rec = mpsfm_rec
        self.correspondences = correspondences

        self.depth_cons_thresh = self.conf.depth_cons_thresh
        self.reg_batch_dc_times_failed = 0
        self.cons_thresh_times_increased = 0
        self.skip_dc_check = False

    def at_registration_success(self):
        """Reset depth consistency variables after successful registration."""
        self.log("Resetting depth consistency variables", level=2)
        self.cons_thresh_times_increased = 0
        self.depth_cons_thresh = self.conf.depth_cons_thresh
        self.reg_batch_dc_times_failed = 0
        for imid in self.mpsfm_rec.images:
            self.mpsfm_rec.images[imid].ignore_matches_AP = {}
            self.mpsfm_rec.images[imid].failed_dc_check = False
        self.skip_dc_check = False

    def relax_thresholds(self):
        """Relax depth consistency thresholds after failed registration."""
        self.log("Relaxing depth consistency thresholds", level=1)
        self.depth_cons_thresh *= 1.3
        self.cons_thresh_times_increased += 1
        self.reg_batch_dc_times_failed = 0
        for imid in self.mpsfm_rec.images:
            self.mpsfm_rec.images[imid].ignore_matches_AP = {}
            self.mpsfm_rec.images[imid].failed_dc_check = False
        self.log(f"\tNew depth consistency threshold: {self.depth_cons_thresh}", level=1)
        self.log(f"\tCons thresh times increased: {self.cons_thresh_times_increased}", level=1)
        self.log(f"\tDepth consistency failed {self.reg_batch_dc_times_failed} times", level=1)

    def find_min_buffer(self, Dab, Pab, buffer_shapes):
        """Find the minimum distance buffer for depth consistency check."""
        min_dabs_out = np.full(buffer_shapes, np.inf)
        Pab_x, Pab_y = np.array(Pab).T

        # Update minimum distances and corresponding indices
        mask = Dab < min_dabs_out[Pab_y, Pab_x]
        min_dabs_out[Pab_y[mask], Pab_x[mask]] = Dab[mask]

        return min_dabs_out, mask

    def check_depth_consistency(self, imid1, imid2, c=15, score_thresh=None):
        """Check depth consistency between two images."""
        out = self.mpsfm_rec.reproject_depth(imid1, imid2)
        out21 = self.mpsfm_rec.reproject_depth(imid2, imid1)
        out |= {
            rev_key: out21[key]
            for key, rev_key in zip(
                ["depth1", "p2D12", "depth12", "mask12", "valid1_mask"],
                ["depth2", "p2D21", "depth21", "mask21", "valid2_mask"],
            )
        }

        isdepth1_in_canv = out["mask12"].reshape(out["valid1_mask"].shape)
        isdepth2_in_canv = out["mask21"].reshape(out["valid2_mask"].shape)

        p2D12 = out["p2D12"][out["mask12"]].astype(int)
        p2D21 = out["p2D21"][out["mask21"]].astype(int)

        p2D12 = out["p2D12"][isdepth1_in_canv].astype(int)
        p2D21 = out["p2D21"][isdepth2_in_canv].astype(int)
        minbuffer12, mask12buffer_ = self.find_min_buffer(
            out["depth12"][isdepth1_in_canv], p2D12, isdepth2_in_canv.shape
        )
        mask12buffer = np.zeros(isdepth1_in_canv.shape, dtype=bool)
        mask12buffer[isdepth1_in_canv] = mask12buffer_
        minbuffer21, mask21buffer_ = self.find_min_buffer(
            out["depth21"][isdepth2_in_canv], p2D21, isdepth1_in_canv.shape
        )
        mask21buffer = np.zeros(isdepth2_in_canv.shape, dtype=bool)
        mask21buffer[isdepth2_in_canv] = mask21buffer_
        if score_thresh is None:
            score_thresh = self.conf.depth_cons_valid_thresh

        var1 = self.mpsfm_rec.images[imid1].depth.uncertainty.copy()
        var1 /= self.mpsfm_rec.images[imid1].depth.conf.prior_std_multiplier**2
        y, x = np.where(mask12buffer)
        keypoints1 = np.array([x, y]).T
        covs1 = self.mpsfm_rec.lifted_pointcovs_cam(
            out["depth1"][mask12buffer], self.mpsfm_rec.camera(imid1), keypoints1, var1[mask12buffer]
        )
        covs1w = self.mpsfm_rec.rotate_covs_to_world(covs1, imid1)
        covs12 = self.mpsfm_rec.rotate_covs_to_cam(covs1w, imid2)
        std1 = var1**0.5
        std1bar = covs12[:, 2, 2] ** 0.5

        var2 = self.mpsfm_rec.images[imid2].depth.uncertainty.copy()
        var2 /= self.mpsfm_rec.images[imid2].depth.conf.prior_std_multiplier**2
        y, x = np.where(mask21buffer)
        keypoints2 = np.array([x, y]).T
        covs2 = self.mpsfm_rec.lifted_pointcovs_cam(
            out["depth2"][mask21buffer], self.mpsfm_rec.camera(imid2), keypoints2, var2[mask21buffer]
        )
        covs2w = self.mpsfm_rec.rotate_covs_to_world(covs2, imid2)
        covs21 = self.mpsfm_rec.rotate_covs_to_cam(covs2w, imid1)
        std2 = var2**0.5
        std2bar = covs21[:, 2, 2] ** 0.5
        t1 = minbuffer12[p2D12[:, 1], p2D12[:, 0]] - out["depth2"][p2D12[:, 1], p2D12[:, 0]]
        t1 /= ((std1bar * c) ** 2 + (std2[p2D12[:, 1], p2D12[:, 0]] * c) ** 2) ** 0.5
        t2 = minbuffer21[p2D21[:, 1], p2D21[:, 0]] - out["depth1"][p2D21[:, 1], p2D21[:, 0]]
        t2 /= ((std2bar * c) ** 2 + (std1[p2D21[:, 1], p2D21[:, 0]] * c) ** 2) ** 0.5

        surface1 = np.abs(t1) < score_thresh
        occl1 = t1 > score_thresh
        invalid1 = t1 < -score_thresh
        surface2 = np.abs(t2) < score_thresh
        occl2 = t2 > score_thresh
        invalid2 = t2 < -score_thresh

        valid1 = np.zeros(isdepth1_in_canv.shape, dtype=bool)
        valid2 = np.zeros(isdepth2_in_canv.shape, dtype=bool)

        valid1[isdepth1_in_canv] = surface1 + occl1
        valid2[isdepth2_in_canv] = surface2 + occl2
        occl1_ = np.zeros(isdepth1_in_canv.shape, dtype=bool)
        occl2_ = np.zeros(isdepth2_in_canv.shape, dtype=bool)
        occl1_[isdepth1_in_canv] = occl1
        occl2_[isdepth2_in_canv] = occl2
        invalid1_ = np.zeros(isdepth1_in_canv.shape, dtype=bool)
        invalid2_ = np.zeros(isdepth2_in_canv.shape, dtype=bool)
        invalid1_[isdepth1_in_canv] = invalid1
        invalid2_[isdepth2_in_canv] = invalid2
        surface1_ = np.zeros(isdepth1_in_canv.shape, dtype=bool)
        surface2_ = np.zeros(isdepth2_in_canv.shape, dtype=bool)
        surface1_[isdepth1_in_canv] = surface1
        surface2_[isdepth2_in_canv] = surface2

        return {
            "valid1": valid1,
            "valid2": valid2,
            "occl1": occl1_,
            "occl2": occl2_,
            "invalid1": invalid1_,
            "invalid2": invalid2_,
            "surface1": surface1_,
            "surface2": surface2_,
            "valid1_mask": isdepth1_in_canv,
            "valid2_mask": isdepth2_in_canv,
        }

    def init_pair(self, init_pair):
        """Check if the initial pair of images is valid based on depth consistency."""
        ref_imid = list(init_pair)[0]
        score_thresh = self.conf.init_valid_thresh
        out = self.check_bundle_depth_concistency(ref_imid, {"optim_ids": init_pair}, score_thresh=score_thresh)
        success = out[0] <= self.conf.init_depth_cons_thresh
        return success

    def pre_fail(self, imid):
        """
        Check if the image should be failed before registration based on previous depth consistency.
        """
        # have we checked for depth consistency before?
        last_dc_score = self.mpsfm_rec.images[imid].last_dc_score
        if last_dc_score is None:
            return False

        # have we tried resampling inliers?
        times_inliers_resampled = self.mpsfm_rec.images[imid].dc_times_inliers_resampled
        if self.conf.depth_consistency_resample and times_inliers_resampled == 0:
            return False

        # have we failed too many times and should force registration?
        if self.skip_dc_check:
            return False

        # Final skip logic
        raise NotImplementedError("This should be implemented")
        if last_dc_score > self.depth_cons_thresh:
            print(f"Depth consistency prefailed for {imid}: {last_dc_score}")
            self.reg_batch_dc_times_failed += 1
            return True
        return False

    def at_failure(self, imid):
        """Handle failure of depth consistency check."""
        image = self.mpsfm_rec.images[imid]
        image.failed_dc_check = True
        if self.conf.depth_consistency_resample:  # exploration
            image.dc_times_inliers_resampled += 1
            print(f"Removing inliers for AP for im {imid}")
            for ref_id, inlier_mask in self.mpsfm_rec.last_ap_inlier_masks.items():
                if len(inlier_mask) > 0:
                    if ref_id in self.mpsfm_rec.images[imid].ignore_matches_AP:
                        used = ~self.mpsfm_rec.images[imid].ignore_matches_AP[ref_id]
                        self.mpsfm_rec.images[imid].ignore_matches_AP[ref_id][used] |= inlier_mask
                    else:
                        self.mpsfm_rec.images[imid].ignore_matches_AP[ref_id] = inlier_mask
            self.reg_batch_dc_times_failed += 1
        else:
            self.reg_batch_dc_times_failed += 1

    def check_image(self, imid, bundle):
        """Check depth consistency of an image in local bundle."""
        score, _ = self.check_bundle_depth_concistency(imid, bundle)

        if score > self.depth_cons_thresh:
            print(f"\nDepth consistency failed for {imid}: {score}!!!")
            self.at_failure(imid)
            return False
        self.log(f"Depth consistency passed for {imid}: {score}", level=1)
        return True

    def check_bundle_depth_concistency(self, imid, bundle, score_thresh=None):
        """Check depth consistency of a bundle of images."""
        self.log(f"Checking depth consistency of {imid}: {self.mpsfm_rec.images[imid].name}...", level=2)
        optim_ids = list(bundle["optim_ids"] - {imid})
        collect = {}
        for ref_imid in optim_ids:
            collect[ref_imid] = self.check_depth_consistency(imid, ref_imid, score_thresh=score_thresh)
        ref_notvalid = [~v["valid2"] * v["valid2_mask"] for _, v in collect.items()]
        ref_mask = [v["valid2_mask"] * ~v["occl2"] for _, v in collect.items()]
        qry_notvalid = [~v["valid1"] * v["valid1_mask"] for _, v in collect.items()]
        qry_mask = [v["valid1_mask"] * ~v["occl1"] for _, v in collect.items()]
        ref_sum_not_valids = [np.sum(notvalid) for notvalid in ref_notvalid]
        ref_sum_valids = [np.sum(valid) for valid in ref_mask]
        qry_sum_not_valids = [np.sum(notvalid) for notvalid in qry_notvalid]
        qry_sum_valids = [np.sum(valid) for valid in qry_mask]
        tot_ref_im_ratios = np.sum(ref_sum_not_valids) / (np.sum(ref_sum_valids).clip(0.1))
        tot_qry_im_ratios = np.sum([qry_sum_not_valids]) / (np.sum(qry_sum_valids).clip(0.1))

        max_ratios = np.max([tot_ref_im_ratios, tot_qry_im_ratios])
        return max_ratios, (
            sum(np.sum(v["valid1_mask"]) for v in collect.values()),
            sum(np.sum(v["valid2_mask"]) for v in collect.values()),
        )
