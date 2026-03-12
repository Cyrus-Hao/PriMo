import cv2
import numpy as np

from mpsfm.utils import viz_3d
from mpsfm.utils.geometry import unproject_depth_map_to_world
from mpsfm.utils.io import read_image


class VisualizationUtils:
    """Visualization utils mixin for reconstruction."""

    def vis_depth_maps(
        self,
        images_dir,
        dmap_rescale=0.5,
        fig=None,
        prior=False,
        m=None,
        name=None,
        input_masks=None,
        **kwargs,
    ):
        if m is None:
            m = {"valid", "metric_scale", "continuity", "outlier"}
        if fig is None:
            fig = viz_3d.init_figure()

        cmaps = {imid: read_image(images_dir / self.images[imid].name) for imid in self.registered_images}
        cmaps = {
            imid: cv2.resize(
                cmaps[imid],
                None,
                fx=dmap_rescale * self.images[imid].camera.sx,
                fy=dmap_rescale * self.images[imid].camera.sy,
                interpolation=cv2.INTER_NEAREST,
            )
            for imid in cmaps
        }

        if len(m) > 0:
            reg_imids = self.registered_images.keys()
            masks = {imid: np.ones(self.images[imid].depth.data.shape, dtype=bool) for imid in reg_imids}
            if "valid" in m:
                masks = {imid: (masks[imid] * self.images[imid].depth.valid) for imid in reg_imids}
            if "metric_scale" in m:
                masks = {
                    imid: (masks[imid] * ((self.images[imid].depth.data / self.images[imid].depth.scale) < 100))
                    for imid in reg_imids
                }
            if "continuity" in m:
                masks = {imid: (masks[imid] * self.images[imid].depth.continuity_mask) for imid in reg_imids}
            if "outlier" in m:
                masks = {
                    imid: (
                        masks[imid] * ((self.images[imid].depth.data) < (6 * np.median(self.images[imid].depth.data)))
                    )
                    for imid in reg_imids
                }
            if input_masks is not None:
                masks = {imid: (input_masks[imid] * masks[imid]) for imid in reg_imids}
            masks = {
                imid: cv2.resize(
                    masks[imid].astype(float),
                    None,
                    fx=dmap_rescale,
                    fy=dmap_rescale,
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                for imid in masks
            }
        else:
            masks = {}
        shown_names = set()
        for imid, image in self.registered_images.items():
            camera = self.cameras[image.camera_id]
            depth = self.images[imid].depth.data_prior if prior else self.images[imid].depth.data

            if dmap_rescale is not None:
                depth = cv2.resize(
                    depth,
                    None,
                    fx=dmap_rescale,
                    fy=dmap_rescale,
                    interpolation=cv2.INTER_NEAREST,
                )

            H = image.cam_from_world.inverse().matrix()
            H = np.vstack((H, np.array([0, 0, 0, 1])))
            K = camera.calibration_matrix()
            K[:2, :] *= dmap_rescale * self.images[imid].camera.sx
            mask = masks.get(imid, None)
            points_3d = unproject_depth_map_to_world(depth, K, H, mask=mask)

            c = cmaps[imid]
            c = c[mask] if mask is not None else c.flatten()

            ds = 1.5
            use_name = image.name if name is None else name
            viz_3d.plot_points(fig, points_3d, color=c, ps=ds, name=use_name, showlegend=use_name not in shown_names)
            shown_names.add(use_name)

        return fig

    def vis_cameras(self, fig=None, name=None, size=1, color="rgb(255, 0, 0)", **kwargs):
        if fig is None:
            fig = viz_3d.init_figure()
        viz_3d.plot_cameras(fig, self.rec, name=name, size=size, color=color, **kwargs)
        return fig

    def visualization(self, cmaps=None, masks=None, prior=False, fig=None, **kwargs):
        """Visualizes reconstruction with input RGB images."""

        fig = self.vis_depth_maps(cmaps=cmaps, masks=masks, prior=prior, fig=fig, **kwargs)
        fig = self.vis_colmap_reconstruction(fig, **kwargs)
        return fig

    def vis_colmap_points(self, fig=None, name=None, mode=None, ps=4, **kwargs):
        """Visualizes colmap reconstruction"""
        if fig is None:
            fig = viz_3d.init_figure()

        def blue_to_red_colormap(value):
            if value == -1:
                green = 255
                red = 255
                blue = 255
            else:
                blue = int(value * 255)
                red = int((1 - value) * 255)
                green = 0
            return (red, green, blue, 1)

        cols = kwargs.pop("color", None)
        if cols is None:
            cols = "rgba(255, 0, 0, 0.5)"
        if mode is None:
            mode = (
                "sparse_mask" if "sparse" in self.conf.matches_mode and "dense" in self.conf.matches_mode else "uncert"
            )

        if mode == "uncert":
            cols = []
            for pt3D_id in self.points3D:
                if pt3D_id in self.point_covs.data:
                    var = np.trace(self.point_covs.data[pt3D_id])  # [2,2]
                    cols.append(1 / (var**0.5))
                else:
                    cols.append(-1)
            cols = np.array(cols)
            cols[np.isnan(cols)] = 0.001
            cols = [f"rgba{blue_to_red_colormap((col/(np.median(cols))).clip(0,1))}" for col in cols]
        elif mode == "sparse_mask":
            is_sparse = {}
            is_sparse = {
                imidA: np.zeros(len(imageA.points2D), dtype=bool) for imidA, imageA in self.registered_images.items()
            }
            is_sparse = {
                imidA: self.correspondences.sparse_im_masks[imageA.name]
                for imidA, imageA in self.registered_images.items()
            }

            cols = []

            for p3d in self.points3D.values():
                track = [(track.image_id, track.point2D_idx) for track in p3d.track.elements]
                pts_arse_sparse = [is_sparse[imid][pt2did] for imid, pt2did in track]
                cols.append(~np.all(pts_arse_sparse))
            cols = [f"rgba{blue_to_red_colormap(1-col)}" for col in cols]

        else:
            cols = []
            print(len(self.rec.points3D))
            for pt3D_id in self.rec.points3D:

                cols.append(self.rec.points3D[pt3D_id].track.length())
            cols = np.array(cols, dtype=float)
            print(cols.min(), cols.max(), np.median(cols), np.mean(cols))
            s4mask = cols < 4
            l4mask = cols >= 4
            cols[s4mask] = 0
            cols[l4mask] = 1
            cols = [f"rgba{blue_to_red_colormap(1-col)}" for col in cols]
        viz_3d.plot_reconstruction(
            fig,
            self.rec,
            color=cols,
            name=(name if name else "map"),
            points_rgb=False,
            ps=ps,
            cameras=False,
            **kwargs,
        )

        return fig
